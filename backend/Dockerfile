FROM python:3.13-slim

# opencv-python-headless still needs these at runtime for image codec / video I/O support.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# ultralytics downloads yolov8n.pt on first use if it isn't already present.
ENV YOLO_MODEL_PATH=yolov8n.pt

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
