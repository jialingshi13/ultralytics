# Free-friendly image: Flask host only. The browser runs MediaPipe.
FROM python:3.12-slim-bookworm

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY index.html server.py score.js ./

ENV POSTURE_INFER=client \
    POSTURE_PRELOAD=0

EXPOSE 10000
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 60 server:app
