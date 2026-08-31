import os
import io
import time
import shutil
from typing import Optional

from fastapi import (
    FastAPI,
    Request,
    UploadFile,
    File,
    Form,
    HTTPException,
)
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# --------------------------------------------------
# Config & Directories
# --------------------------------------------------
load_dotenv()

MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024  # 200 MB limit

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

for folder in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    os.makedirs(folder, exist_ok=True)

# --------------------------------------------------
# Whisper Model Management (Lightweight ~75MB - ~145MB)
# --------------------------------------------------
# Using faster-whisper with int8 quantization keeps memory & model files strictly < 200MB.
whisper_model = None
whisper_available = False

try:
    from faster_whisper import WhisperModel
    print("Loading lightweight Whisper 'base' model (CPU int8, ~140MB)...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    whisper_available = True
    print("Whisper 'base' model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load faster-whisper: {e}")
    whisper_available = False

# Fallback Google Speech Recognition if faster-whisper fails
try:
    import speech_recognition as sr
    sr_available = True
except ImportError:
    sr_available = False

# Fallback OpenAI API client if configured
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
API_KEY = OPENAI_API_KEY or OPENROUTER_API_KEY or GROQ_API_KEY
openai_client = None

if API_KEY:
    try:
        from openai import OpenAI
        if API_KEY.startswith("sk-or-") or OPENROUTER_API_KEY:
            openai_client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
        elif API_KEY.startswith("gsk_") or GROQ_API_KEY:
            openai_client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
        else:
            openai_client = OpenAI(api_key=API_KEY)
    except Exception as err:
        openai_client = None

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(
    title="Audio to Text Converter",
    description="Convert audio files to text files quickly and locally with lightweight Whisper model (under 200MB).",
    version="2.1.0",
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# Transcribe Helper
# --------------------------------------------------
def run_whisper_transcription(file_path: str, language: Optional[str] = None):
    """Transcribe audio using local faster-whisper."""
    if not whisper_model:
        raise RuntimeError("Whisper model is not initialized.")

    lang_param = None
    if language and language.lower() not in ("auto", "auto-detect", ""):
        lang_param = language.lower().split("-")[0]

    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=5,
        language=lang_param,
        vad_filter=True,
    )

    text_parts = [segment.text.strip() for segment in segments]
    full_text = " ".join(text_parts).strip()
    
    if not full_text:
        full_text = "[No clear speech detected in audio]"

    return {
        "text": full_text,
        "language": getattr(info, "language", language or "unknown"),
        "language_probability": round(getattr(info, "language_probability", 1.0), 2),
        "duration": round(getattr(info, "duration", 0.0), 2),
    }

def run_google_sr_fallback(file_path: str) -> str:
    """Fallback using SpeechRecognition library."""
    if not sr_available:
        raise RuntimeError("SpeechRecognition fallback is not available.")
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = r.record(source)
        return r.recognize_google(audio_data)

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def home(request: Request):
    """Render the simple audio-to-text conversion UI."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "max_size_mb": MAX_FILE_SIZE_MB,
            "engine_status": "Local Whisper (Ready)" if whisper_available else ("Cloud API (Ready)" if openai_client else "Offline Fallback"),
            "model_size": "< 200 MB (Optimized)",
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "online",
        "whisper_local": whisper_available,
        "max_upload_size_mb": MAX_FILE_SIZE_MB,
        "model": "Whisper base int8 (~140MB)",
    }

@app.post("/convert")
@app.post("/transcribe")
async def convert_audio_to_text(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
):
    """Convert an uploaded audio file (up to 200MB) directly into a text file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file was selected.")

    allowed_exts = {
        ".mp3", ".mpeg", ".mpg", ".mpga", ".wav", ".m4a", ".ogg", ".flac",
        ".webm", ".aac", ".mp4", ".wma", ".opus", ".mka", ".3gp", ".amr", ".caf"
    }
    
    ext = os.path.splitext(file.filename)[1].lower()
    if not ext:
        ext = ".wav"

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(allowed_exts)}",
        )

    clean_stem = "".join(c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in ("-", "_", " ")).strip()
    if not clean_stem:
        clean_stem = "audio_converted"

    timestamp = int(time.time())
    temp_audio_name = f"{clean_stem}_{timestamp}{ext}"
    temp_audio_path = os.path.join(UPLOAD_DIR, temp_audio_name)

    start_time = time.time()
    total_bytes = 0

    try:
        # Stream file to disk and enforce 200MB limit
        with open(temp_audio_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_MB}MB.",
                    )
                buffer.write(chunk)

        transcribed_text = ""
        detected_lang = language or "auto"
        engine_used = "Whisper Base (Local int8)"

        # 1. Try local Whisper
        if whisper_available:
            try:
                res = run_whisper_transcription(temp_audio_path, language=language)
                transcribed_text = res["text"]
                detected_lang = res["language"]
            except Exception as we:
                print(f"Whisper transcription failed: {we}")

        # 2. Try Google fallback
        if not transcribed_text and sr_available and ext == ".wav":
            try:
                transcribed_text = run_google_sr_fallback(temp_audio_path)
                engine_used = "Google Web Speech"
            except Exception as ge:
                print(f"Google speech fallback failed: {ge}")

        # 3. Try Cloud API fallback
        if not transcribed_text and openai_client:
            try:
                with open(temp_audio_path, "rb") as af:
                    c_res = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af,
                    )
                transcribed_text = c_res.text
                engine_used = "Cloud Whisper API"
            except Exception as ce:
                print(f"Cloud API transcription failed: {ce}")

        if not transcribed_text:
            raise HTTPException(
                status_code=500,
                detail="Could not extract text from the audio. Please verify the audio has clear speech.",
            )

        # Save to Text (.txt) File
        txt_filename = f"{clean_stem}_{timestamp}.txt"
        txt_path = os.path.join(OUTPUT_DIR, txt_filename)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)

        processing_duration = round(time.time() - start_time, 2)
        words = len(transcribed_text.split())
        chars = len(transcribed_text)
        file_size_mb = round(total_bytes / (1024 * 1024), 2)

        return {
            "success": True,
            "text": transcribed_text,
            "txt_filename": txt_filename,
            "download_url": f"/download/{txt_filename}",
            "stats": {
                "detected_language": str(detected_lang).upper(),
                "word_count": words,
                "char_count": chars,
                "file_size_mb": file_size_mb,
                "process_time_sec": processing_duration,
                "engine": engine_used,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled error in convert_audio_to_text: {e}")
        raise HTTPException(status_code=500, detail=f"Conversion error: {str(e)}")

    finally:
        # Clean up temporary uploaded audio
        if os.path.exists(temp_audio_path):
            try:
                os.remove(temp_audio_path)
            except Exception:
                pass

@app.get("/download/{filename}")
async def download_text_file(filename: str):
    """Download the converted text (.txt) file."""
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Text file not found.")

    return FileResponse(
        path=file_path,
        media_type="text/plain; charset=utf-8",
        filename=safe_name,
        headers={"Content-Disposition": f'attachment; filename="{safe_name}"'}
    )