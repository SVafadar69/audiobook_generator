#!/usr/bin/env python
# coding: utf-8

# In[9]:

# def token_counter(texts: str, system_prompt: str) -> int:
#     """Return the combined token count of system_prompt + texts for gpt-4o encoding."""
#     enc = tiktoken.encoding_for_model("gpt-4o")
#     return len(enc.encode(system_prompt) + enc.encode(texts))


book = '/Users/sv/Downloads/ai_warface.epub'
import zipfile, os

with zipfile.ZipFile(book, "r") as z:
    z.extractall("epub_contents")


# In[10]:

### CONSTANTS
from dotenv import load_dotenv
load_dotenv()
import zipfile
import os
from openai import OpenAI, AsyncOpenAI
import nest_asyncio
nest_asyncio.apply()
import re
import subprocess
import time
from bs4 import BeautifulSoup 
import requests
book = '/Users/sv/Downloads/ai_warface.epub'
rapidai_api_key = os.getenv('rapidai_api_key')
openai_api_key = os.getenv('openai_api_key')

#import tiktoken


def parse_response(user_input: str):
    """
    Deconstruct user query to extract book_name, author, and file_type.
    Returns list: [book_name, author, file_type]
    Uses fastest OpenAI model available in 2026: gpt-5.4-nano
    """
    # Initialize client with API key from environment variable
    client = OpenAI(api_key=openai_api_key)
    
    # System prompt with proper formatting - fixed quote escaping
    system_prompt = '''Your job is to deconstruct the given user query and return it as a list. 
    The query should contain book_name, author, and file_type. 
    Your response should be as: [book_name, author, file_type]. 
    Return only that list - no json strings, no thinking tags, etc.
    If no author or file type is provided, use "" as the variable. 
    
    User query: {user_input}
    Your response: '''
    
    # Format the prompt correctly using .format() instead of f-string with replace
    formatted_prompt = system_prompt.format(user_input=user_input)
    
    # Use the fastest generation model: gpt-5.4-nano (based on 2026 benchmarks)
    response = client.responses.create(
        model="gpt-5.4-nano",
        input=formatted_prompt
    )
    
    # Extract and return just the response text
    return response.output_text.strip()

def load_api_key():
    openai_api_key = os.getenv('openai_api_key')
    rapidai_api_key = os.getenv('rapidai_api_key')
    return openai_api_key, rapidai_api_key

def retrieve_book(book_name: str, author: str, file_type: list):

    print(f'file type: {type(file_type)}')
    print(f'{file_type}')
    if not author: 
        author = ""
    
    if isinstance(file_type, str):
        file_type = [file_type] if file_type else []
    file_type = [f for f in file_type if f]

    if len(file_type) > 1: file_type = ','.join(file_type)
    elif len(file_type) == 1: file_type = file_type[0]
    else: file_type = ""

    non_member_sources = {'lgli', 'lgrs', 'nexusstc'}
    
    url = "https://annas-archive-api.p.rapidapi.com/search"

    print(f'using filetype: {file_type}')


    querystring = {"q":f"{book_name}","author":f"{author}","cat":"fiction, nonfiction, comic, magazine, musicalscore, other, unknown","page":"1","ext":f"{file_type}","sort":"mostRelevant","source":"libgenLi, libgenRs"}

    headers = {
	"x-rapidapi-key": f"{rapidai_api_key}",
	"x-rapidapi-host": "annas-archive-api.p.rapidapi.com",
	"Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers, params=querystring)
    books = response.json().get('books', [])


    if len(books): 
        extracted = sorted(
            [
                {
                    'title':  book['title'],
                    'author': book['author'],
                    'md5':    book['md5'],
                    'imgUrl': book['imgUrl'],
                    'year':   book['year'], 
                }
                for book in books
                if non_member_sources.intersection(book.get('sources', []))
            ],
            key=lambda x: x['author']
        )

    return extracted 

def get_download_url(md5): 
    url = "https://annas-archive-api.p.rapidapi.com/download"

    querystring = {"md5":f"{md5}"}

    headers = {
        "x-rapidapi-key": f"{rapidai_api_key}",
        "x-rapidapi-host": "annas-archive-api.p.rapidapi.com",
        "Content-Type": "application/json"
    }

    try: 
        response = requests.get(url, headers=headers, params=querystring)
        response.raise_for_status()

        return response.json()
    except Exception as e: 
        return f'Error received when trying to download the book: {e}\n \
            Please try another title.'

def download_book(download_url, book_name): 
    try: 
        r = requests.get(download_url, stream=True)
        with open(book_name, 'wb') as f: 
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f'Successfully downloaded: {book_name}!')
    except Exception as e: 
        print(f'Error when downloading book from m5: {e}')

def get_epub_contents(book, output_path):
    with zipfile.ZipFile(book, "r") as z:
        z.extractall(output_path)
        return output_path



import re


# In[3]:


chapters_index = '/Users/sv/Desktop/audiobooks/epub_contents/OEBPS/xhtml'
chapters_dir = os.listdir(chapters_index)
chapters = [chapter for chapter in chapters_dir if '_ch' in chapter]
print(f'chapters: {chapters}')


# In[6]:


from bs4 import BeautifulSoup 

chapter_text = ''
output_dir = os.getcwd()
for filename in sorted(chapters):
    with open(os.path.join(chapters_index, filename)) as file:
        soup = BeautifulSoup(file, 'html.parser')
        text = soup.get_text(separator='\n').strip()
    with open(os.path.join(output_dir, filename.replace('.xhtml', '.txt')), 'w') as out: 
        out.write(text)


# In[7]:


def clean_text(chapter_text): 
    with open(chapter_text, 'r', encoding='utf-8') as file: 
        text = file.read()
        # numbers on their own line!
        # can also just do simple LLM call
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\s+', ' ', text).strip()
    return text




# In[ ]:


import os 
# 1) get number index between ch and .txt 
# 2) clean the text, use that number index as chapter name 
# 3) look through each to see if anything is missing
chapter_indices = os.listdir(".")
print(f'chapter indices: {chapter_indices}')
chapter_indices = sorted([chapter_index.split('.')[0].split('ch')[-1] for chapter_index in chapter_indices])
chapter_indices = sorted([int(x) for x in chapter_indices if x.isdigit()])


# In[ ]:


chapters = [chapter for chapter in os.listdir('.') if chapter.endswith('.txt')]


# In[24]:


sorted_chapter_files = sorted(chapters, key = lambda x: int(x.split('_ch')[1].split('.txt')[0]))
print(sorted_chapter_files)


# In[25]:


def clean_all_chapters(sorted_chapter_files, chapter_indices):
    for index, chapter in zip(chapter_indices, sorted_chapter_files):
        cleaned_text = clean_text(chapter)
        with open(f'chapter_{index}.txt', 'w', encoding='utf-8') as file:
            file.write(cleaned_text)
        os.remove(chapter)

clean_all_chapters(sorted_chapter_files, chapter_indices = chapter_indices)


# In[ ]:


''''''
from openai import OpenAI 
api_key = ''
client = OpenAI(api_key = api_key)


# In[4]:


import os
def get_char_count(chapters): 
    for chapter in chapters: 
        chapter_text = open(chapter, 'r', encoding = 'utf-8').read()
        print(len(chapter_text))

chapters = sorted([chapter for chapter in os.listdir('.') if chapter.endswith('.txt')], 
    key = lambda x: int(x.replace('chapter_', '').replace('.txt', '')))
get_char_count(chapters)


# In[5]:


def trim_chapter(chapter: str, max_chunk_size: int = 4096) -> list[str]:
    words = chapter.split()
    chunks = []
    current_chunk = []
    current_length = 0 

    for word in words: 
        word_length = len(word) + 1 # why is there +1 
        if current_length + word_length > max_chunk_size: # explain
            chunks.append(' '.join(current_chunk)) # total current chunk - all words 
            current_chunk = [word] # explain -> this resets? 
            current_length = word_length # explain
        else:
            current_chunk.append(word) # adding words if under max
            current_length += word_length # adding word length - this is 4096

    if current_chunk: # when does the loop exit? -> if did not go over 4096, exists in void
        chunks.append(' '.join(current_chunk))

    return chunks


# In[6]:


# mk audio dir for each chapter? 
def trim_all_chapter(chapter_text): 
    chapter_chunks = trim_chapter(chapter_text)
    return chapter_chunks 


# In[80]:


os.makedirs("audiobook", exist_ok=True)


# In[7]:


ordered_chapter_texts = sorted([chapter for chapter in os.listdir('.') if chapter.endswith('.txt')],
    key = lambda x: int(x.replace('chapter_', '').replace('.txt', '')))
ordered_chapter_indices = sorted([int(chapter.replace('chapter_', '').replace('.txt', '')) for chapter in ordered_chapter_texts])
print(ordered_chapter_indices)


# In[ ]:


# split into chapters
# split chapters into chunks 
# convert each chunk -> audio 
# concatenate in order of audio chunk (based on chapter)
# save audio to respective folder - dir audio_book 
# make a chapter, in each chapter store the audio file 
# or concatenate all audio snippets into one large chapter 

import time
def chapter_to_audio(chapter_indices, chapters_texts): 
    start_time = time.time()
    for index, chapter in zip(chapter_indices, chapters_texts): 
        chapter_dir = f'audiobook/chapter_{index}'
        os.makedirs(chapter_dir, exist_ok=True)
        chapter_text = open(chapter, 'r', encoding='utf-8').read()
        chapter_chunks = trim_all_chapter(chapter_text)
        for chunk_index, chunk in enumerate(chapter_chunks): 
            response = client.audio.speech.create(
                model = 'tts-1', 
                voice = 'alloy', 
                input = chunk
            )
            response.stream_to_file(f'{chapter_dir}/chunk_{chunk_index}.mp3')
    print(F'one chapter took: {time.time() - start_time}')


# In[ ]:


chapter_to_audio(chapter_indices[3:], ordered_chapter_texts[3:])


# In[9]:


# should be same time as 
import nest_asyncio
nest_asyncio.apply()
import os

from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=api_key)


# In[14]:


# try to generate entire audiobook at once - 
# everything should be async
# 1.5 mins across everything
# everything should be async - will be bottlenecked by AI generation time/costs
import time

async def generate_audiobook(
    chapter_indices: list[int], 
    chapter_texts: list[str], ) -> None: 
    start_time = time.time()
    for index, chapter in zip(chapter_indices, chapter_texts): 
        chapter_dir = f'audiobook/chapter_{index}'
        os.makedirs(chapter_dir, exist_ok = True) 
        chapter_text = open(chapter, 'r', encoding='utf-8').read()
        chapter_chunks = trim_all_chapter(chapter_text)

        all_tasks = [
            _save_chunk(chapter_dir, chunk_index, chunk)
            for chunk_index, chunk in enumerate(chapter_chunks)
        ]
        try: 
            await asyncio.gather(*all_tasks)
            print(f"Chapter {index} — all {len(chapter_chunks)} chunk(s) done in one batch.")
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

        print(f"Chapter {index} complete — {len(chapter_chunks)} chunk(s) total.")
        print(f'Chapter takes {time.time() - start_time} to generate the entire chapter')

async def _save_chunk(chapter_dir: str, chunk_index: int, chunk: str) -> None:
    """Generate and save a single audio chunk."""
    response = await client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=chunk,
    )
    output_path = f"{chapter_dir}/chunk_{chunk_index}.mp3"
    response.stream_to_file(output_path)

async def main() -> None:
    await generate_audiobook(ordered_chapter_indices[5:], ordered_chapter_texts[5:])


if __name__ == "__main__":
    asyncio.run(main())


# In[ ]:


import os
import re
import sys
import subprocess
import tempfile


def find_chapters(root):
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


# In[45]:


import subprocess
cur_dir = os.getcwd()
audiobook_dir = cur_dir + '/audiobook'
os.makedirs('complete_audiobook', exist_ok=True)
chapters = sorted(os.listdir(audiobook_dir),key=lambda x: int(re.search(r'\d+', x).group()))
print(f'chapters: {chapters}')
final_output_path = os.getcwd() + '/final_audiobook.mp3'
chapter_audiobooks_folder = os.path.join(os.getcwd(), 'complete_audiobook')


# In[46]:


def create_chapter_audiobooks(chapters): 
    for chapter in chapters: 
        chunks = [chunk for chunk in os.listdir(os.path.join(audiobook_dir, chapter))]
        sorted_chunks = sorted(chunks, key = lambda x: int(re.search(r'\d+', x).group()))
        final_chunks = [os.path.join(audiobook_dir, chapter, sorted_chunk) for sorted_chunk in sorted_chunks]
        concat_str = '|'.join(final_chunks); print(f'concat string: {concat_str}')
        chapter_output_path = os.path.join(f"{chapter_audiobooks_folder}/{chapter}_audiobook.mp3")
        command = ['ffmpeg', '-i', f'concat:{concat_str}','-acodec','copy',chapter_output_path]
        subprocess.run(command)


# In[47]:


create_chapter_audiobooks(chapters)


# In[ ]:


def merge_all_chapters_into_final_book(audiobook_chapters): 
    cur_audiobooks = [book for book in os.listdir(audiobook_chapters)]
    sorted_audiobooks = sorted(cur_audiobooks, key = lambda x: int(re.search(r'\d+', x).group()))
    final_sorted_audiobooks = [os.path.join(chapter_audiobooks_folder, audiobook) for audiobook in \
        sorted_audiobooks]
    concat_str = '|'.join(final_sorted_audiobooks)
    print(f'concat_str: {concat_str}')
    final_audiobook_output_path = os.getcwd() + '/final_audiobook.mp3'
    command = ['ffmpeg', '-i', f'concat:{concat_str}','-acodec','copy',final_audiobook_output_path]
    subprocess.run(command)


# In[58]:


merge_all_chapters_into_final_book(chapter_audiobooks_folder)


# In[ ]:


# existing textbooks 
# existing books 
# article compilations -> APIs* -> Audiobook text -> LLMs (rearrange)
# RAG over existing data(?)*
# podcasts
# start of chapter 1, end of chapter 1 


# In[135]:


import datetime 
from exa_py import Exa
from dotenv import load_dotenv 
load_dotenv()
import os 

exa = Exa(api_key=os.getenv('exa_api_key'))



# In[136]:


from datetime import date

def make_exa_call(
    query: str = "", 
    num_results: int = 100,  
    start_published_date="2025-09-01",
    end_published_date=str(datetime.date.today()),
    _type: str = 'instant'
):
    # Switch to search_and_contents to retrieve page data
    results = exa.search_and_contents(
        query=query, 
        num_results=num_results,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        type=_type,
        summary = True,
        text = True
        
    )

    return results


# In[212]:


from datetime import date

def make_exa_call(
    query: str = "", 
    num_results: int = 30,  
    start_published_date="2025-09-01",
    end_published_date=str(datetime.date.today()),
    _type: str = 'instant'
    #_type: str = 'deep'
):
    # Switch to search_and_contents to retrieve page data
    results = exa.search_and_contents(
        query=query, 
        num_results=num_results,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        type=_type,
        summary = True,
        text = True
        
    )

    return results


# In[213]:


start_time = time.time()
response = make_exa_call(query="How do defense companies get started in Canada?")
print(f'took: {time.time() - start_time}')


# In[ ]:


# 18 seconds for 30 deep articles 
# 7 seconds for 30 instant articles
# 120k tokens
# summary over 30 is impossible 
# after clean: 


# In[215]:


texts = [result.text for result in response.results]
texts[0]


# In[219]:


total_tokens = 0 
for text in texts: 
    token_count = token_counter(system_prompt = system_prompt, texts = text)
    total_tokens += token_count

print(f'total_tokens: {total_tokens}')


# In[ ]:


# 30 tokens total 


# In[208]:


with open('final_output.txt', 'w', encoding = 'utf-8') as file: 
    file.write(texts[0])

# wait 10 seconds (groq w/ smaller results - articles or summaries), wait 1.5 mins, 
# Q: scenario: you have 1-2 hours on the treadmill, 3 hours 
# q: you need direct knowledge (highest quality info)
# highlights/highlightscores -> relevance per excerpt; category (research paper); high-quality domains
# rewrite query; deep agentic search


# In[209]:


token_count = token_counter(system_prompt = "", texts = texts[0])
print(token_count)


# In[ ]:


from datetime import date

def make_exa_call_summary(
    query: str = "", 
    num_results: int = 30,  
    start_published_date="2025-09-01",
    end_published_date=str(datetime.date.today()),
    _type: str = 'instant'
):
    # Switch to search_and_contents to retrieve page data
    results = exa.search_and_contents(
        query=query, 
        num_results=num_results,
        start_published_date=start_published_date,
        end_published_date=end_published_date,
        type=_type,
        summary = True
        
    )

    return results

response = make_exa_call_summary(query="How do defense companies get started in Canada?", num_results=30)


# In[183]:


texts = [result.text for result in response.results]
print(len(texts))


# In[184]:


texts


# In[ ]:


from openai import OpenAI
from dotenv import load_dotenv 
load_dotenv()
openai_api_key = os.getenv('openai_api_key')

def organize_text(texts: list[str] = None):
    '''What do you need this AI to do? -> '''
    if not texts: return ""
    system_prompt = '''You are an AI tasked at taking different corpuses of text and arranging 
    them in a way that is linearly relevant. Do not omit information that is important (random details)
    or facts is okay to omit. Just arrange the text so that it is linear/semantically grouped by 
    similarity. You will receive a list of strings as input - return one giant string as your response, 
    that is the the text arranged. Return only the arranged text as your response - no json tags, no thinking strings, etc. 
    '''
    formatted_texts = '\n'.join(f"-{t}" for t in texts)
    user_message = f"<text>\n{formatted_texts}\n</text>"
    client = OpenAI(api_key = openai_api_key)
    response = client.responses.create(
        model="gpt-5-nano",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ],
    )
    print(response)

    return response.output_text

# either use summaries (and more volume), less results (), more results no cleaning()



# In[223]:


start_time = time.time()
text = organize_text(texts)
token_count = token_counter(system_prompt = system_prompt, texts = text)
print(f'time took: {time.time() - start_time}')
print(f'total cleaned tokens: {token_count}')


# In[ ]:


# 1 with no cleaning 
# 1 with cleaning 
# 


# In[193]:


import tiktoken

def token_counter(texts, system_prompt):
    enc = tiktoken.encoding_for_model("gpt-4o")  # or "gpt-4", "gpt-3.5-turbo"
    
    tokens = enc.encode(system_prompt) + enc.encode(texts)
    return len(tokens)


# In[176]:


from groq import Groq
groq_api_key = os.getenv('groq_api_key')
system_prompt = '''You are an AI tasked at taking different corpuses of text and arranging 
    them in a way that is linearly relevant. Do not omit information that is important (random details)
    or facts is okay to omit. Just arrange the text so that it is linear/semantically grouped by 
    similarity. You will receive a list of strings as input - return one giant string as your response, 
    that is the the text arranged. Return only the arranged text as your response - no json tags, no thinking strings, etc. 
    '''

formatted_texts = '\n'.join(f"-{t}" for t in texts)
token_count = token_counter(formatted_texts, system_prompt)
print(token_count)


# In[ ]:


def organize_text_groq(texts):
    if not texts: return "You need to supply texts of articles"
    
    formatted_texts = '\n'.join(f"-{t}" for t in texts)
    user_message = f"<text>\n{formatted_texts}\n</text>"
    client = Groq(api_key = groq_api_key)
    completion = client.chat.completions.create(
        model="qwen/qwen3-32b",
        messages=[
      {
        "role": "system",
        "content": f"{system_prompt}"
      },
      {
        "role": "user",
        "content": f"{user_message}"
      }
    ],
        temperature=0.2,
        max_completion_tokens=4096,
        top_p=0.95,
        reasoning_effort="default"
    )

    return completion.choices[0].message.content


# In[210]:


import time 
from dotenv import load_dotenv
load_dotenv()
start_time = time.time()
cleaned_texts = organize_text_groq(texts)
print(f'cleaned_texts: {cleaned_texts}') # 
print(f'time took: {time.time() - start_time}')


# In[143]:


cleaned_texts = organize_text(texts)
print(f'cleaned_texts: {cleaned_texts}') # 


# In[ ]:




