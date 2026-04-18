# ── Standard library ──────────────────────────────────────────────────────────
import asyncio
import datetime
import os
import re
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import date

# ── Third-party ────────────────────────────────────────────────────────────────
import nest_asyncio
import tiktoken
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from exa_py import Exa
from groq import Groq
from openai import AsyncOpenAI, OpenAI

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

openai_api_key = os.getenv("openai_api_key")
groq_api_key   = os.getenv("groq_api_key")
exa_api_key    = os.getenv("exa_api_key")

# ── Clients ────────────────────────────────────────────────────────────────────
sync_openai_client  = OpenAI(api_key=openai_api_key)
async_openai_client = AsyncOpenAI(api_key=openai_api_key)
exa                 = Exa(api_key=exa_api_key)

# ── Prompts ────────────────────────────────────────────────────────────────────
ORGANIZE_SYSTEM_PROMPT = (
    "You are an AI tasked at taking different corpuses of text and arranging "
    "them in a way that is linearly relevant. Do not omit information that is "
    "important (random details or facts is okay to omit). Just arrange the text "
    "so that it is linear/semantically grouped by similarity. You will receive a "
    "list of strings as input - return one giant string as your response, that is "
    "the text arranged. Return only the arranged text as your response - no json "
    "tags, no thinking strings, etc."
)


# ── EPUB extraction ────────────────────────────────────────────────────────────

def extract_epub(book_path: str, output_dir: str = "epub_contents") -> None:
    """Unzip an epub archive into output_dir."""
    with zipfile.ZipFile(book_path, "r") as z:
        z.extractall(output_dir)


def parse_chapters(chapters_index: str, output_dir: str | None = None) -> None:
    """
    Convert xhtml chapter files found in chapters_index into plain-text .txt files
    written to output_dir (defaults to cwd).
    """
    if output_dir is None:
        output_dir = os.getcwd()

    chapters_dir = os.listdir(chapters_index)
    chapters = [c for c in chapters_dir if "_ch" in c]

    for filename in sorted(chapters):
        with open(os.path.join(chapters_index, filename)) as file:
            soup = BeautifulSoup(file, "html.parser")
            text = soup.get_text(separator="\n").strip()
        out_path = os.path.join(output_dir, filename.replace(".xhtml", ".txt"))
        with open(out_path, "w") as out:
            out.write(text)


# ── Text cleaning ──────────────────────────────────────────────────────────────

def clean_text(chapter_path: str) -> str:
    """Read a chapter file, strip lone page numbers, and collapse whitespace."""
    with open(chapter_path, "r", encoding="utf-8") as file:
        text = file.read()
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_all_chapters(sorted_chapter_files: list[str], chapter_indices: list[int]) -> None:
    """
    Clean every chapter file and save it as chapter_<N>.txt,
    removing the original xhtml-derived file afterward.
    """
    for index, chapter in zip(chapter_indices, sorted_chapter_files):
        cleaned_text = clean_text(chapter)
        with open(f"chapter_{index}.txt", "w", encoding="utf-8") as file:
            file.write(cleaned_text)
        os.remove(chapter)


# ── Chapter utilities ──────────────────────────────────────────────────────────

def get_char_count(chapters: list[str]) -> None:
    """Print character count for each chapter file."""
    for chapter in chapters:
        text = open(chapter, "r", encoding="utf-8").read()
        print(len(text))


def find_chapters(root: str) -> list[tuple[int, str]]:
    """
    Scan root for subdirectories matching 'chapter_<N>' (case-insensitive).
    Returns a sorted list of (chapter_num, chapter_dir_path).
    """
    chapter_pattern = re.compile(r"^chapter_(\d+)$", re.IGNORECASE)
    chapters = []
    for entry in os.scandir(root):
        if not entry.is_dir():
            continue
        m = chapter_pattern.match(entry.name)
        if m:
            chapters.append((int(m.group(1)), entry.path))
    return sorted(chapters, key=lambda x: x[0])


# ── Audio chunking ─────────────────────────────────────────────────────────────

def trim_chapter(chapter: str, max_chunk_size: int = 4096) -> list[str]:
    """Split a chapter string into word-safe chunks of at most max_chunk_size chars."""
    words = chapter.split()
    chunks: list[str] = []
    current_chunk: list[str] = []
    current_length = 0

    for word in words:
        word_length = len(word) + 1  # +1 accounts for the space separator
        if current_length + word_length > max_chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def trim_all_chapter(chapter_text: str) -> list[str]:
    """Convenience wrapper around trim_chapter."""
    return trim_chapter(chapter_text)


# ── Synchronous TTS ────────────────────────────────────────────────────────────

def chapter_to_audio(chapter_indices: list[int], chapters_texts: list[str]) -> None:
    """Convert chapter text files to mp3 chunks using the synchronous OpenAI TTS API."""
    start_time = time.time()
    for index, chapter in zip(chapter_indices, chapters_texts):
        chapter_dir = f"audiobook/chapter_{index}"
        os.makedirs(chapter_dir, exist_ok=True)
        chapter_text = open(chapter, "r", encoding="utf-8").read()
        chapter_chunks = trim_all_chapter(chapter_text)
        for chunk_index, chunk in enumerate(chapter_chunks):
            response = sync_openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=chunk,
            )
            response.stream_to_file(f"{chapter_dir}/chunk_{chunk_index}.mp3")
    print(f"All chapters took: {time.time() - start_time:.1f}s")


# ── Asynchronous TTS ───────────────────────────────────────────────────────────

async def _save_chunk(chapter_dir: str, chunk_index: int, chunk: str) -> None:
    """Generate and save a single audio chunk asynchronously."""
    response = await async_openai_client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=chunk,
    )
    response.stream_to_file(f"{chapter_dir}/chunk_{chunk_index}.mp3")


async def generate_audiobook(
    chapter_indices: list[int],
    chapter_texts: list[str],
) -> None:
    """
    Async audiobook generator. Attempts to fire all chunks for a chapter
    concurrently; falls back to batches of 10 on failure.
    """
    start_time = time.time()
    for index, chapter in zip(chapter_indices, chapter_texts):
        chapter_dir = f"audiobook/chapter_{index}"
        os.makedirs(chapter_dir, exist_ok=True)
        chapter_text = open(chapter, "r", encoding="utf-8").read()
        chapter_chunks = trim_all_chapter(chapter_text)

        all_tasks = [
            _save_chunk(chapter_dir, chunk_index, chunk)
            for chunk_index, chunk in enumerate(chapter_chunks)
        ]
        try:
            await asyncio.gather(*all_tasks)
            print(f"Chapter {index} — all {len(chapter_chunks)} chunk(s) done.")
        except Exception as e:
            print(f"Chapter {index} — full batch failed ({e}), falling back to batches of 10.")
            for batch_start in range(0, len(chapter_chunks), 10):
                batch = [
                    _save_chunk(chapter_dir, chunk_index, chunk)
                    for chunk_index, chunk in enumerate(
                        chapter_chunks[batch_start : batch_start + 10],
                        start=batch_start,
                    )
                ]
                await asyncio.gather(*batch)
                print(f"Chapter {index} — fallback batch {batch_start}–{batch_start + len(batch) - 1} done.")

        print(f"Chapter {index} complete — {len(chapter_chunks)} chunk(s), {time.time() - start_time:.1f}s elapsed.")


# ── ffmpeg concatenation ───────────────────────────────────────────────────────

def create_chapter_audiobooks(
    chapters: list[str],
    audiobook_dir: str,
    chapter_audiobooks_folder: str,
) -> None:
    """Concatenate per-chunk mp3s into one mp3 per chapter via ffmpeg."""
    for chapter in chapters:
        chunks = os.listdir(os.path.join(audiobook_dir, chapter))
        sorted_chunks = sorted(chunks, key=lambda x: int(re.search(r"\d+", x).group()))
        final_chunks = [os.path.join(audiobook_dir, chapter, c) for c in sorted_chunks]
        concat_str = "|".join(final_chunks)
        chapter_output_path = os.path.join(chapter_audiobooks_folder, f"{chapter}_audiobook.mp3")
        subprocess.run(["ffmpeg", "-i", f"concat:{concat_str}", "-acodec", "copy", chapter_output_path])


def merge_all_chapters_into_final_book(
    chapter_audiobooks_folder: str,
    final_output_path: str | None = None,
) -> None:
    """Merge all per-chapter mp3s into a single final audiobook file via ffmpeg."""
    if final_output_path is None:
        final_output_path = os.path.join(os.getcwd(), "final_audiobook.mp3")

    cur_audiobooks = os.listdir(chapter_audiobooks_folder)
    sorted_audiobooks = sorted(cur_audiobooks, key=lambda x: int(re.search(r"\d+", x).group()))
    final_sorted = [os.path.join(chapter_audiobooks_folder, a) for a in sorted_audiobooks]
    concat_str = "|".join(final_sorted)
    subprocess.run(["ffmpeg", "-i", f"concat:{concat_str}", "-acodec", "copy", final_output_path])


# ── Exa search ─────────────────────────────────────────────────────────────────

def make_exa_call(
    query: str = "",
    num_results: int = 30,
    start_published_date: str = "2025-09-01",
    end_published_date: str = str(datetime.date.today()),
    _type: str = "instant",
):
    """Search and retrieve full text via Exa."""
    return exa.search_and_contents(
        query=query,
        num_results=num_results,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        type=_type,
        summary=True,
        text=True,
    )


def make_exa_call_summary(
    query: str = "",
    num_results: int = 30,
    start_published_date: str = "2025-09-01",
    end_published_date: str = str(datetime.date.today()),
    _type: str = "instant",
):
    """Search and retrieve summaries only via Exa."""
    return exa.search_and_contents(
        query=query,
        num_results=num_results,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        type=_type,
        summary=True,
    )


# ── Token counting ─────────────────────────────────────────────────────────────

def token_counter(texts: str, system_prompt: str) -> int:
    """Return the combined token count of system_prompt + texts for gpt-4o encoding."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    return len(enc.encode(system_prompt) + enc.encode(texts))


# ── Text organisation ──────────────────────────────────────────────────────────

def organize_text(texts: list[str]) -> str:
    """Arrange a list of text corpuses into a single linearly coherent string via OpenAI."""
    if not texts:
        return ""
    formatted_texts = "\n".join(f"-{t}" for t in texts)
    user_message = f"<text>\n{formatted_texts}\n</text>"
    response = sync_openai_client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": ORGANIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )
    return response.output_text


def organize_text_groq(texts: list[str]) -> str:
    """Arrange a list of text corpuses into a single linearly coherent string via Groq."""
    if not texts:
        return "You need to supply texts of articles"
    formatted_texts = "\n".join(f"-{t}" for t in texts)
    user_message = f"<text>\n{formatted_texts}\n</text>"
    client = Groq(api_key=groq_api_key)
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
            {"role": "system", "content": ORGANIZE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
        max_completion_tokens=4096,
        top_p=0.95,
        reasoning_effort="default",
    )
    return completion.choices[0].message.content


# ── Entry point ────────────────────────────────────────────────────────────────

async def main(
    chapter_indices: list[int] | None = None,
    chapter_texts: list[str] | None = None,
) -> None:
    """
    Generate the full audiobook asynchronously.
    If chapter_indices / chapter_texts are not supplied, they are derived
    from .txt files in the current working directory.
    """
    nest_asyncio.apply()

    if chapter_texts is None:
        chapter_texts = sorted(
            [c for c in os.listdir(".") if c.endswith(".txt")],
            key=lambda x: int(x.replace("chapter_", "").replace(".txt", "")),
        )
    if chapter_indices is None:
        chapter_indices = sorted(
            int(c.replace("chapter_", "").replace(".txt", "")) for c in chapter_texts
        )

    await generate_audiobook(chapter_indices, chapter_texts)


if __name__ == "__main__":
    asyncio.run(main())