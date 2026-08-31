import os
import io
import time
from typing import Optional, Union, BinaryIO

from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from starlette.concurrency import run_in_threadpool
# pyrefly: ignore [missing-import]
from dotenv import load_dotenv

# --------------------------------------------------
# Config & Directories
# --------------------------------------------------
load_dotenv()

MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "output"
STATIC_DIR = "static"
TEMPLATES_DIR = "templates"

for folder in [UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR, TEMPLATES_DIR]:
    os.makedirs(folder, exist_ok=True)

# --------------------------------------------------
# Whisper Model Setup
# --------------------------------------------------
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", str(os.cpu_count() or 4)))

whisper_model = None
whisper_available = False

try:
    # pyrefly: ignore [missing-import]
    from faster_whisper import WhisperModel
    print(f"Loading Whisper '{WHISPER_MODEL_NAME}' ({WHISPER_COMPUTE_TYPE}, threads={WHISPER_THREADS})...")
    whisper_model = WhisperModel(
        WHISPER_MODEL_NAME,
        device="cpu",
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_THREADS,
        num_workers=1,
    )
    whisper_available = True
    print("Whisper model loaded successfully!")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------
app = FastAPI(title="Ultra-Fast Speech to Text")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --------------------------------------------------
# Transcribe Function (Synchronous Worker)
# --------------------------------------------------
def run_whisper_transcription(audio_source: Union[str, BinaryIO], language: Optional[str] = "en"):
    """Greedy decoding with VAD silence removal."""
    if not whisper_model:
        raise RuntimeError("Whisper model is not initialized.")

    segments, info = whisper_model.transcribe(
        audio_source,
        language="en" if "en" in WHISPER_MODEL_NAME else (language or None),
        beam_size=1,
        best_of=1,
        temperature=0,
        condition_on_previous_text=False,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=300, speech_pad_ms=200),
        without_timestamps=True,
    )

    text_parts = [segment.text.strip() for segment in segments if segment.text]
    full_text = " ".join(text_parts).strip()

    return {
        "text": full_text or "[No clear speech detected in audio]",
        "language": getattr(info, "language", "en"),
        "language_probability": round(getattr(info, "language_probability", 1.0), 2),
        "duration": round(getattr(info, "duration", 0.0), 2),
    }

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "max_size_mb": MAX_FILE_SIZE_MB,
            "engine_status": "Local Whisper (Ultra-Fast)" if whisper_available else "Offline Fallback",
            "model_size": f"< 150 MB ({WHISPER_MODEL_NAME})",
        }
    )

@app.get("/health")
async def health():
    return {
        "status": "online",
        "whisper_local": whisper_available,
        "model": f"{WHISPER_MODEL_NAME} {WHISPER_COMPUTE_TYPE}",
        "threads": WHISPER_THREADS,
    }

@app.post("/convert")
@app.post("/transcribe")
async def convert_audio_to_text(
    file: UploadFile = File(...),
    language: Optional[str] = Form("en"),
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No audio file selected.")

    start_time = time.time()
    
    # Read directly into RAM
    audio_bytes = await file.read()
    total_bytes = len(audio_bytes)
    
    if total_bytes > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=f"File exceeds limit of {MAX_FILE_SIZE_MB}MB.")

    audio_buffer = io.BytesIO(audio_bytes)
    
    # Execute non-blocking CPU transcription in threadpool
    try:
        res = await run_in_threadpool(run_whisper_transcription, audio_buffer, language)
        transcribed_text = res["text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")

    # Save output text file
    clean_stem = "".join(c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in ("-", "_", " ")).strip() or "transcription"
    timestamp = int(time.time())
    txt_filename = f"{clean_stem}_{timestamp}.txt"
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(transcribed_text)

    processing_duration = round(time.time() - start_time, 2)

    return {
        "success": True,
        "text": transcribed_text,
        "txt_filename": txt_filename,
        "download_url": f"/download/{txt_filename}",
        "stats": {
            "detected_language": res["language"].upper(),
            "word_count": len(transcribed_text.split()),
            "char_count": len(transcribed_text),
            "file_size_mb": round(total_bytes / (1024 * 1024), 2),
            "process_time_sec": processing_duration,
            "engine": f"Whisper {WHISPER_MODEL_NAME} ({WHISPER_COMPUTE_TYPE})",
        }
    }

@app.get("/download/{filename}")
async def download_text_file(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=file_path, media_type="text/plain; charset=utf-8", filename=safe_name)