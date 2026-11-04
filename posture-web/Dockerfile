# CPU image for the quality path. Render web services have no GPU.
FROM python:3.12-slim-bookworm

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-server.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-server.txt \
    && python -c "import torch; assert torch.version.cuda is None, torch.__version__; from ultralytics import YOLO; YOLO('yolo26l-pose.pt')"

COPY index.html server.py score.js ./

ENV POSTURE_INFER=server \
    POSTURE_MODEL=yolo26l-pose.pt \
    POSTURE_IMGSZ=640 \
    POSTURE_PRELOAD=1

EXPOSE 10000
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 server:app
