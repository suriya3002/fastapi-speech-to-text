# 🎙️ Ultra-Optimized FastAPI + faster-whisper (Low-Memory CPU Architecture)

A high-throughput, memory-efficient Speech-to-Text service built with **FastAPI** and **faster-whisper (CTranslate2)**. Specifically engineered for resource-constrained platforms (shared 1 vCPU, 512MB–1GB RAM) running bare-metal on the host system without Docker.

---

## ⚡ Key Performance Engineering Features

1. **Strict < 200MB RAM Footprint**:
   - Uses `int8` quantization (`compute_type="int8"`) with `tiny.en` (~45–75MB RAM) or `base.en` (~140–180MB RAM).
   - Pruned dependencies to strip heavy audio/cloud libraries and avoid memory leaks.
2. **Pure In-Memory Audio Pipeline**:
   - Direct `io.BytesIO` payload processing: Audio bytes are received asynchronously in RAM and decoded in memory by `faster_whisper` without touching disk storage.
3. **Non-Blocking Async Event Loop**:
   - Heavy CTranslate2 CPU operations are wrapped inside `starlette.concurrency.run_in_threadpool`, keeping FastAPI's `asyncio` event loop responsive.
4. **Decoded for Speed (Greedy + VAD)**:
   - `beam_size=1`, `best_of=1`, `temperature=0.0`, `condition_on_previous_text=False`
   - `vad_filter=True` (min silence: 300ms, speech padding: 200ms) with `without_timestamps=True`.
5. **Thread Contention Prevention**:
   - Enforces `OMP_NUM_THREADS=1` and `CT2_USE_EXPERIMENTAL_PACKED_GEMM=1` to eliminate multi-threaded context switching overhead on 1 vCPU instances.
6. **Zero Cold-Start Lag**:
   - Pre-warming script downloads weights and initializes CTranslate2 GEMM kernels at build time.

---

## 🚀 Quick Start (Native Host Setup)

### 1. Set Up Environment & Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
.\venv\Scripts\activate

# Install pruned dependencies
pip install -r requirements.txt
```

### 2. Pre-Warm Model (Build Time / Step 1)

Pre-download model weights and warm up CPU inference to eliminate startup latency:

```bash
python preload_model.py
```

### 3. Launch Native Production Server

Run Uvicorn with optimized single-worker loop and httptools parser:

#### Linux / macOS (Recommended)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --loop uvloop --http httptools --no-access-log
```

#### Windows
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1 --http httptools --no-access-log
```

Open your browser at **[http://localhost:8000](http://localhost:8000)** or check **[http://localhost:8000/health](http://localhost:8000/health)**.

---

## ⚙️ Environment Variables (`.env`)

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL` | `base.en` | Model size: `tiny.en` (<80MB RAM) or `base.en` (<180MB RAM) |
| `WHISPER_COMPUTE_TYPE` | `int8` | Strictly `int8` quantization for minimal memory |
| `WHISPER_THREADS` | `1` | CPU threads for inference (1–2 for 1 vCPU) |
| `OMP_NUM_THREADS` | `1` | OpenMP threads to avoid thread thrashing |
| `CT2_USE_EXPERIMENTAL_PACKED_GEMM` | `1` | Enables CTranslate2 experimental packed matrix multiplication |
| `MAX_FILE_SIZE_MB` | `25` | Upload ceiling to prevent memory spikes on 512MB RAM |

---

## 📡 API Endpoints

- `GET /` - Web UI for file conversion.
- `GET /health` - Returns JSON health status, model type, compute mode, and thread limits.
- `POST /convert` or `POST /transcribe` - Accepts `file: UploadFile` and optional `language: str`, returns JSON with transcribed text, runtime stats, and download link.
- `GET /download/{filename}` - Downloads saved `.txt` transcription.

---

## 📂 Project Structure

```
fastapi-speech-to-text/
├── main.py              # Refactored async FastAPI app + in-memory Whisper pipeline
├── preload_model.py     # Build-time model pre-warm script
├── requirements.txt     # Pruned minimal dependency list
├── .env.example         # Production environment template
├── templates/
│   └── index.html       # Web UI
├── static/
│   ├── app.js           # Lightweight frontend controller
│   └── style.css        # Responsive styling
└── output/              # Output .txt exports
```

---

## 📄 License
MIT License
