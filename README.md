# 🎙️ FastAPI Speech-to-Text Studio

A modern, high-performance Speech-to-Text web application built with **FastAPI**, **faster-whisper (Whisper Base Multilingual)**, and **Web Speech API**. 

Supports **99+ languages**, auto-language detection, speech translation to English, live real-time dictation, and audio file processing with zero API keys or cloud costs required.

---

## ✨ Features

- 🧠 **Local Whisper Base Multilingual Model**: Runs completely offline on CPU with `int8` quantization using CTranslate2.
- 🌐 **99+ Languages**: Automatic language detection with accuracy confidence scores.
- 🔄 **Transcribe & Translate**: Transcribe audio in its native language or translate directly into English text.
- 📁 **File Upload**: Drag-and-drop audio transcription for `.mp3`, `.wav`, `.m4a`, `.webm`, `.ogg`, `.flac`.
- 🎙️ **Microphone Recording**: In-browser audio recorder with animated HTML5 Canvas waveform visualizer.
- ⚡ **Live Real-Time Dictation**: Instantaneous streaming speech recognition directly in the browser via Web Speech API.
- 📝 **Transcript Studio**: Live word/character counts, one-click copy to clipboard, and export to `.txt` and `.json`.
- 🛡️ **Zero-Cost Fallbacks**: Includes Google Speech Recognition and Cloud Whisper API options.

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
