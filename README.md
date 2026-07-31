# 🎧 Free Audiobook Generator

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)

An AI-powered application that instantly transforms any book title or complex research theme into a polished, high-quality audiobook. Whether you want to listen to a classic novel or a synthesized briefing on cutting-edge geopolitical topics, this tool automates the retrieval, aggregation, and Text-to-Speech (TTS) conversion pipeline.

---

## 🚀 How to Structure Your Project for GitHub

When publishing a project on GitHub to ensure visitors instantly know how to run, understand, and contribute to it, standard repository conventions are used. A professional GitHub repository typically includes the following core files:

1. **`README.md`** (This file): The primary landing page explaining what the project does, prerequisites, installation steps, and usage instructions.
2. **`requirements.txt` / `pyproject.toml`**: Lists all Python dependencies so users can install them with a single command.
3. **`.gitignore`**: Prevents unnecessary files (like virtual environments, cache files, and downloaded audio outputs) from being tracked in git.
4. **`.env.example`**: A template file showing users which environment variables (API keys) are required.
5. **`main.py` or `app.py`**: The entry point for running the application.

---

## ✨ Features

- **Book Mode**: Automatically searches and downloads target books (e.g., via Anna's Archive integration) and prepares them for speech synthesis.
- **Theme/Research Mode**: Executes deep searches across the web using the **Exa API**, aggregates relevant articles, summarizes them into a coherent narrative script, and converts them to audio.
- **Natural Voice Synthesis**: Integrates advanced TTS models to generate expressive, natural-sounding audio files.

---

## 📌 Roadmap & Things Left to Do

- [ ] **UI Enhancements**: Create an interactive download button directly in the user interface to easily export generated audiobooks (`.mp3` / `.m4b`).
- [ ] **Loading States & Feedback**: Implement an animated loading screen and real-time progress indicators in the UI during long-running retrieval and synthesis tasks.
- [ ] **GPU Acceleration**: Migrate the heavy inference and TTS backend to support CUDA-enabled GPUs, drastically reducing generation time.

---

## 🛠️ Prerequisites

Before running the project locally, ensure you have the following installed on your system:
- **Python 3.10 or higher**
- **FFmpeg** (required for audio manipulation and stitching)
- **NVIDIA CUDA Toolkit** (optional, recommended for GPU acceleration)

---

## 📦 Installation & Setup

Follow these step-by-step instructions to set up the project on your local machine.

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/free-audiobook-generator.git
cd free-audiobook-generator
```

### 2. Create and Activate a Virtual Environment
```bash
# On macOS/Linux
python3 -m venv venv
source venv/bin/activate

# On Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory based on the provided `.env.example`:
```env
EXA_API_KEY=your_exa_api_key_here
OUTPUT_DIR=./outputs
TTS_MODEL=coqui/XTTS-v2
```

---

## 💻 Running the Project

To launch the application interface (e.g., Streamlit or Gradio):

```bash
python app.py
```
*(Or if using Streamlit)*:
```bash
streamlit run app.py
```

Once running, open your browser and navigate to `http://localhost:7860` (or the port specified in your terminal). 

1. Enter a book name (e.g., *Moby Dick*) or a research theme (e.g., *Defense spending in Ukraine*).
2. Click **Generate Audiobook**.
3. Once processing completes, use the download button to save your file.

---

## 🗂️ Typical Git File Structure

A clean, production-ready repository layout looks like this:

```text
free-audiobook-generator/
├── .github/
│   └── workflows/        # CI/CD pipelines (optional)
├── assets/               # Screenshots, banners, demo audio
├── outputs/              # Generated audiobooks (ignored in git)
├── .env.example          # Template for environment variables
├── .gitignore            # Files to ignore (venv, .env, outputs/)
├── README.md             # Project documentation & instructions
├── requirements.txt      # Python dependencies
├── app.py                # Main UI / Entry point
└── src/
    ├── __init__.py
    ├── downloader.py     # Anna's Archive & Exa API integration
    └── tts_engine.py     # Text-to-speech conversion logic
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an issue for bug fixes, feature requests (like the CUDA backend or UI loading screen), or performance enhancements.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.