Free Audiobook Generator

Generate audiobooks from books, documents, or researched articles.

Enter a book title, upload a supported document, or provide a topic such as:

"Defense spending in Ukraine"

The application can:

Convert user-provided ebooks and documents into audiobooks
Find public-domain or legally accessible books
Research a topic using the Exa API
Convert documents, articles and books into human-sounding audiobooks - entirely for free. 
Export the generated audio as a WAV file

Project Status

This project is currently under development.

Remaining Work

Add an audiobook download button to the user interface

Add a loading and progress screen

Move inference to a GPU backend

Add CUDA acceleration

Improve error handling

Add support for additional document formats

---

Requirements
Python 3.10 or newer
An Exa API key
Kokoro TTS model files
FFmpeg
A CUDA-capable GPU for optional accelerated inference
Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY

Create a virtual environment:

python -m venv .venv

Activate the environment.

macOS or Linux
source .venv/bin/activate
Windows PowerShell
.venv\Scripts\Activate.ps1

Install the Python dependencies:

pip install -r requirements.txt


Place your associated .env keys in the .env file. You will need the following: 
exa_api_key 
groq_api_key 

Place the Kokoro model files in the expected model directory:

models/
├── kokoro-v1.0.onnx
└── kokoro-v1.0.bin
Running the Project

Replace the command below with the actual entry-point filename used by the project:


streamlit run app.py


After starting the application, open the local address printed in the terminal.

For example:

http://127.0.0.1:8000

---



Project Structure
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── kokoro-v1.0.onnx
│   └── kokoro-v1.0.bin
├── output/
├── src/
│   ├── research.py
│   ├── text_processing.py
│   ├── text_to_speech.py
│   └── audio_processing.py
└── ui/
