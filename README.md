# 🎙️ FastAPI Speech-to-Text Studio (English Ultra-Fast)

A modern, high-performance Speech-to-Text web application built with **FastAPI**, **faster-whisper (English-Only Whisper Base.en)**, and lightweight CPU `int8` quantization. 

Specially optimized for **maximum speed and low memory** on cloud hosted platforms (Render, Railway, HuggingFace Spaces, Fly.io, Cloud Run, AWS) and local machines.

---

## ✨ Features

- ⚡ **English-Only Whisper `base.en` Model**: Specialized English-only weights that are 2-3x faster and more accurate on English audio than multilingual models.
- 🚀 **Hosted Platform CPU Acceleration**: Configured with greedy single-path decoding (`beam_size=1`), Silero Voice Activity Detection (VAD), and multi-core CPU thread allocation.
- 📁 **File Upload**: Drag-and-drop audio transcription for `.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`, `.flac` up to 200MB.
- 📝 **Automatic `.txt` File Export**: Instantly transcribes and produces downloadable plain text files.
- 🛡️ **Zero-Cost Fallbacks & Groq Turbo**: Includes Google Speech Recognition fallback and optional Groq/OpenAI cloud fallback (~250x realtime speed).

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/suriya3002/fastapi-speech-to-text.git
cd fastapi-speech-to-text
```

### 2. Create and Activate Virtual Environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
uvicorn main:app --reload
```

Open your browser and navigate to: **[http://localhost:8000](http://localhost:8000)**

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Jinja2, WebSockets
- **AI Models**: `faster-whisper` (OpenAI Whisper Base Multilingual), `SpeechRecognition`, `OpenAI` client
- **Audio Processing**: `soundfile`, `pywav`, `av`
- **Frontend**: HTML5, Vanilla JavaScript, Tailwind CSS, FontAwesome

---

## 📂 Project Structure

```
fastapi-speech-to-text/
├── main.py              # FastAPI server & Whisper inference logic
├── requirements.txt     # Python dependencies
├── .env.example         # Environment template
├── .gitignore           # Git ignore rules
├── templates/
│   └── index.html       # Modern Speech-to-Text Studio UI
├── static/
│   ├── app.js           # Frontend scripts
│   └── style.css        # Stylesheet
├── uploads/             # Temporary uploaded audio storage
├── recordings/          # Temporary voice recordings
└── output/              # Saved transcriptions (.txt)
```

---

## 📄 License
MIT License
