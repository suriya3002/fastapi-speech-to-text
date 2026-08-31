FROM python:3.11-slim-bookworm

WORKDIR /app

# Upgrade base packages to patch security vulnerabilities & install dependencies
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user and necessary writable directories for security compliance
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/uploads /app/output && \
    chown -R appuser:appuser /app

USER appuser

# Multi-threading & CPU environment flags
ENV OMP_NUM_THREADS=4
ENV CT2_USE_EXPERIMENTAL_PACKED_GEMM=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]