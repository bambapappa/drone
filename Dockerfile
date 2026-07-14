FROM python:3.12-slim

WORKDIR /app

# ffmpeg: video decode (files + RTSP); libgl/glib: opencv runtime deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# CPU-only torch first (much smaller than the default CUDA build)
RUN uv pip install --system --no-cache torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Dependencies only at this layer (app/ is copied later for better caching)
COPY pyproject.toml .
RUN uv pip install --system --no-cache -r pyproject.toml

# Bake the detection model into the image so the container runs offline.
ARG MODEL=yolo11n.pt
ENV YOLO_CONFIG_DIR=/data/ultralytics
RUN mkdir -p /data/ultralytics /models /videos && \
    python -c "from ultralytics import YOLO; YOLO('${MODEL}')" && \
    mv ${MODEL} /models/${MODEL}
ENV MODEL=/models/${MODEL} \
    VIDEO_DIR=/videos

COPY app/ ./app/
COPY analysis/ ./analysis/
COPY review/ ./review/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
