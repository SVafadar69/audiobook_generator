# 🎧 Free Audiobook Generator

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-active%20development-orange.svg)

AI-powered audiobook generator. Enter the title of a book, or a concept you want to learn about (i.e. defense spending in Ukraine), and a human-sounding audiobook will be generated for you for free. 

Use the Anna's Archive or Exa API for content curation, and Kokoro-ONNX for TTS generation of the audiobook. 

---

## ✨ Features

- **Book Mode**: Automatically searches and downloads target books (e.g., via Anna's Archive integration) and prepares them for speech synthesis.
- **Theme/Research Mode**: Executes deep searches across the web using the **Exa API**, aggregates relevant articles, summarizes them into a coherent narrative script, and converts them to audio.
- **Natural Voice Synthesis**: Integrates advanced TTS models to generate expressive, natural-sounding audio files.


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
git clone https://github.com/SVafadar69/audiobook_generator.git
cd audiobook-generator
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
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory based on the provided `.env.example`:
```env
exa_api_key=your_exa_api_key_here
groq_api_key=your_groq_api_key_here
```

---

## 💻 Running the Project

To launch the application interface (e.g., Streamlit or Gradio):

```bash
streamlit run app.py
```

Once running, open your browser and navigate to `http://localhost:8501` (or the port specified in your terminal). 

1. Enter a book name (e.g., *Moby Dick*) or a research theme (e.g., *Defense spending in Ukraine*).
2. Click **Generate Audiobook** or press **ENTER**.
3. Once processing completes, the completed audiobook will be `audio.wav` in your local project folder.

---

## 🗂️ Folder Structure

```text
audiobook-generator/
├── models/               # Where the voice + engine files are stored
├── outputs/              # Generated audiobooks (ignored in git)
├── .env                  # Where your .env files are stored
├── README.md             # Project documentation & instructions
├── audio.wav             # This will be your generated audiobook. 
├── requirements.txt      # Python dependencies
├── app.py                # Main UI / Entry point
├── test.py               # Helper functions that contain all relevant audiobook generation functions 
```

---

## 🤝 Contributing

You are welcome to contribute to the project. The remaining needed features are: 
---

## 📌 Roadmap & Things Left to Do

- [ ] **UI Enhancements**: Create a download button in the user interface to export generated audiobooks. Right now they are being written within the project folder locally. 
- [ ] **Loading States & Feedback**: Implement an animated loading screen and real-time progress indicators in the UI during long-running retrieval and synthesis tasks.
- [ ] **GPU Acceleration**: Migrate the heavy inference and TTS backend to support CUDA-enabled GPUs, drastically reducing generation time.
---