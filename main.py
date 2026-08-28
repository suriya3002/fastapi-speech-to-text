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
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
)
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from dotenv import load_dotenv

# --------------------------------------------------
# Whisper Local Multilingual Model Setup
# --------------------------------------------------

whisper_model = None
whisper_available = False

try:
    from faster_whisper import WhisperModel
    print("Loading local Whisper 'base' multilingual model (CPU int8)...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
    whisper_available = True
    print("Local Whisper 'base' multilingual model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load local faster-whisper model: {e}")
    whisper_available = False

# Optional fallback speech recognition
try:
    import speech_recognition as sr
    sr_available = True
except ImportError:
    sr_available = False

try:
    import soundfile as sf
    sf_available = True
except ImportError:
    sf_available = False


# --------------------------------------------------
# Configuration & Provider Setup
# --------------------------------------------------

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CUSTOM_BASE_URL = os.getenv("OPENAI_BASE_URL")

API_KEY = OPENAI_API_KEY or OPENROUTER_API_KEY or GROQ_API_KEY
openai_client = None

if API_KEY:
    try:
        from openai import OpenAI
        if CUSTOM_BASE_URL:
            openai_client = OpenAI(api_key=API_KEY, base_url=CUSTOM_BASE_URL)
        elif API_KEY.startswith("sk-or-") or OPENROUTER_API_KEY:
            openai_client = OpenAI(api_key=API_KEY, base_url="https://openrouter.ai/api/v1")
        elif API_KEY.startswith("gsk_") or GROQ_API_KEY:
            openai_client = OpenAI(api_key=API_KEY, base_url="https://api.groq.com/openai/v1")
        else:
            openai_client = OpenAI(api_key=API_KEY)
    except Exception as err:
        print(f"Cloud OpenAI client init error: {err}")
        openai_client = None


# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="FastAPI Speech To Text",
    description="Local Whisper Base Multilingual Speech-to-Text Studio",
    version="2.0.0",
)


# --------------------------------------------------
# Directories
# --------------------------------------------------

os.makedirs("uploads", exist_ok=True)
os.makedirs("output", exist_ok=True)
os.makedirs("recordings", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)


# --------------------------------------------------
# Static files & Templates
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)

templates = Jinja2Templates(directory="templates")


# --------------------------------------------------
# Routes: Home & Diagnostics
# --------------------------------------------------

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "message": "Speech-to-text server is running",
        "timestamp": time.time(),
        "local_whisper": whisper_available,
    }


@app.get("/api/status")
async def api_status():
    if whisper_available:
        provider = "Local Whisper (Offline)"
        model = "Whisper Base (Multilingual)"
    elif openai_client:
        provider = "Cloud Whisper API"
        model = "whisper-1"
    elif sr_available:
        provider = "Google Speech Recognition"
        model = "Free Web Engine"
    else:
        provider = "Web Speech API"
        model = "Browser Native"

    return {
        "ready": bool(whisper_available or openai_client or sr_available),
        "provider": provider,
        "model": model,
        "engine_type": "local" if whisper_available else "cloud",
        "supported_languages": "99+ Languages (Auto-detect)",
        "fallback_available": sr_available,
    }


# --------------------------------------------------
# Local Whisper & Fallback Transcriber Functions
# --------------------------------------------------

def transcribe_local_whisper(file_path: str, language: Optional[str] = None, task: str = "transcribe"):
    """Transcribe audio locally using the Whisper 'base' multilingual model."""
    if not whisper_model:
        raise RuntimeError("Local Whisper model is not loaded.")

    lang_param = None
    if language and language.lower() not in ("auto", "auto-detect", ""):
        lang_param = language.lower().split("-")[0] # e.g. 'en-US' -> 'en', 'es-ES' -> 'es'

    segments, info = whisper_model.transcribe(
        file_path,
        beam_size=5,
        language=lang_param,
        task=task,
        vad_filter=True, # Voice Activity Detection filters background silence
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    full_text = " ".join(text_parts).strip()
    if not full_text:
        full_text = "[No speech detected in audio]"

    return {
        "text": full_text,
        "language": info.language,
        "language_probability": round(info.language_probability, 2),
        "duration": round(info.duration, 2),
    }


def transcribe_with_google_fallback(file_path: str) -> str:
    """Fallback to free Google Web Speech Recognition."""
    if not sr_available:
        raise RuntimeError("SpeechRecognition module is not installed.")

    ext = os.path.splitext(file_path)[1].lower()
    target_wav_path = file_path
    temp_wav_path = None

    if ext != ".wav" and sf_available:
        try:
            data, samplerate = sf.read(file_path)
            temp_wav_path = os.path.splitext(file_path)[0] + "_fallback.wav"
            sf.write(temp_wav_path, data, samplerate, format="WAV")
            target_wav_path = temp_wav_path
        except Exception as conv_err:
            print(f"soundfile conversion error: {conv_err}")

    r = sr.Recognizer()
    try:
        with sr.AudioFile(target_wav_path) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "[No speech detected in audio]"
    except sr.RequestError as e:
        raise RuntimeError(f"Google Speech Recognition service error: {e}")
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
            except Exception:
                pass


# --------------------------------------------------
# Audio Upload & Transcribe Endpoint
# --------------------------------------------------

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    task: Optional[str] = Form("transcribe"),
):
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No audio file selected.",
        )

    allowed_extensions = {
        ".mp3",
        ".wav",
        ".m4a",
        ".webm",
        ".mp4",
        ".mpeg",
        ".mpga",
        ".ogg",
        ".flac",
    }

    extension = os.path.splitext(file.filename)[1].lower()
    if not extension:
        extension = ".wav"

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format: '{extension}'. Allowed: {', '.join(allowed_extensions)}",
        )

    # Save uploaded audio file
    safe_basename = "".join([c for c in os.path.splitext(file.filename)[0] if c.isalnum() or c in (' ', '-', '_')]).rstrip()
    if not safe_basename:
        safe_basename = f"audio_{int(time.time())}"

    temp_filename = f"{safe_basename}_{int(time.time())}{extension}"
    upload_path = os.path.join("uploads", temp_filename)

    start_time = time.time()

    try:
        with open(upload_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                buffer.write(chunk)

        text = ""
        detected_lang = "auto"
        lang_prob = 1.0
        engine_used = "Local Whisper Base (Multilingual)"

        # 1. PRIMARY: Local Whisper Base Multilingual Model
        if whisper_available:
            try:
                whisper_result = transcribe_local_whisper(
                    upload_path,
                    language=language,
                    task=task or "transcribe",
                )
                text = whisper_result["text"]
                detected_lang = whisper_result["language"]
                lang_prob = whisper_result["language_probability"]
                engine_used = f"Local Whisper Base ({detected_lang.upper()})"
            except Exception as w_err:
                print(f"Local Whisper transcription error: {w_err}")

        # 2. SECONDARY: Google Speech Recognition Fallback
        if not text and sr_available:
            try:
                text = transcribe_with_google_fallback(upload_path)
                engine_used = "Google Speech Fallback (Free)"
            except Exception as g_err:
                print(f"Google speech fallback error: {g_err}")

        # 3. TERTIARY: Cloud OpenAI if configured
        if not text and openai_client:
            try:
                with open(upload_path, "rb") as af:
                    c_res = openai_client.audio.transcriptions.create(
                        model="whisper-1",
                        file=af,
                    )
                text = c_res.text
                engine_used = "Cloud Whisper API"
            except Exception as c_err:
                print(f"Cloud fallback error: {c_err}")

        if not text:
            raise HTTPException(
                status_code=500,
                detail="Could not extract speech from audio file. Please check audio quality.",
            )

        duration = round(time.time() - start_time, 2)
        txt_filename = f"{safe_basename}_{int(time.time())}.txt"
        txt_path = os.path.join("output", txt_filename)

        with open(txt_path, "w", encoding="utf-8") as text_file:
            text_file.write(text)

        word_count = len(text.split()) if text else 0
        char_count = len(text) if text else 0

        return {
            "success": True,
            "filename": txt_filename,
            "text": text,
            "detected_language": detected_lang,
            "language_probability": lang_prob,
            "word_count": word_count,
            "char_count": char_count,
            "duration_seconds": duration,
            "engine": engine_used,
            "download_url": f"/download/{txt_filename}",
        }

    except HTTPException:
        raise
    except Exception as e:
        print("Unexpected transcription exception:", e)
        raise HTTPException(
            status_code=500,
            detail=f"Transcription error: {str(e)}",
        )

    finally:
        # Cleanup uploaded audio file
        if os.path.exists(upload_path):
            try:
                os.remove(upload_path)
            except Exception:
                pass


# --------------------------------------------------
# Download Transcribed Text
# --------------------------------------------------

@app.get("/download/{filename}")
async def download_text(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join("output", safe_filename)

    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail="Transcribed text file not found.",
        )

    return FileResponse(
        path=file_path,
        media_type="text/plain; charset=utf-8",
        filename=safe_filename,
    )


# --------------------------------------------------
# Live WebSocket Streaming
# --------------------------------------------------

@app.websocket("/ws/live")
async def live_transcription(websocket: WebSocket):
    await websocket.accept()
    print("Live WebSocket connected.")

    audio_buffer = bytearray()

    try:
        while True:
            message = await websocket.receive()
            
            if "bytes" in message and message["bytes"]:
                chunk = message["bytes"]
                audio_buffer.extend(chunk)
                
                await websocket.send_json({
                    "type": "audio_ack",
                    "bytes_received": len(audio_buffer),
                    "status": "recording",
                })

            elif "text" in message and message["text"]:
                try:
                    import json
                    payload = json.loads(message["text"])
                    action = payload.get("action")
                    
                    if action == "ping":
                        await websocket.send_json({"type": "pong", "time": time.time()})
                    elif action in ("flush", "finish"):
                        if len(audio_buffer) > 1000:
                            temp_path = os.path.join("recordings", f"ws_{int(time.time())}.wav")
                            with open(temp_path, "wb") as f:
                                f.write(audio_buffer)
                            
                            res_text = ""
                            if whisper_available:
                                try:
                                    w_res = transcribe_local_whisper(temp_path)
                                    res_text = w_res["text"]
                                except Exception as we:
                                    print(f"WS Whisper error: {we}")

                            if not res_text and sr_available:
                                try:
                                    res_text = transcribe_with_google_fallback(temp_path)
                                except Exception:
                                    pass

                            await websocket.send_json({
                                "type": "transcription",
                                "text": res_text or "[Speech processing completed]",
                            })

                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                            audio_buffer.clear()
                        else:
                            audio_buffer.clear()
                except Exception as parse_err:
                    print(f"WS text handle error: {parse_err}")

    except WebSocketDisconnect:
        print("Live WebSocket disconnected.")
    except Exception as e:
        print(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass