"""
Pre-warm and pre-download Whisper model weights at build/deploy time.
Eliminates runtime cold-start lag and validates INT8 quantization on CPU.
"""
import os
import io
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Enforce resource-constrained CPU threading flags
os.environ["OMP_NUM_THREADS"] = os.getenv("OMP_NUM_THREADS", "1")
os.environ["CT2_USE_EXPERIMENTAL_PACKED_GEMM"] = "1"

WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
WHISPER_THREADS = int(os.getenv("WHISPER_THREADS", "1"))

# Safety check for memory-constrained platforms (< 200MB RAM target)
ALLOWED_MODELS = {"tiny.en", "base.en", "tiny", "base"}
if WHISPER_MODEL_NAME not in ALLOWED_MODELS:
    print(f"Warning: '{WHISPER_MODEL_NAME}' might exceed the 200MB RAM budget on constrained instances.")

print(f"--> Pre-downloading Whisper model '{WHISPER_MODEL_NAME}' ({WHISPER_COMPUTE_TYPE})...")

from faster_whisper import WhisperModel

model = WhisperModel(
    WHISPER_MODEL_NAME,
    device="cpu",
    compute_type=WHISPER_COMPUTE_TYPE,
    cpu_threads=WHISPER_THREADS,
    num_workers=1,
)

print("--> Running warm-up inference to prime CTranslate2 GEMM kernels...")
dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence at 16kHz
segments, _ = model.transcribe(
    dummy_audio,
    beam_size=1,
    best_of=1,
    temperature=0.0,
    condition_on_previous_text=False,
    vad_filter=True,
    without_timestamps=True,
)
_ = list(segments)

print("--> Model pre-warming complete! Ready for zero-cold-start production serving.")
