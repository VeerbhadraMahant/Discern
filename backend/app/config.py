import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
YOLO_MODEL_PATH = os.environ.get("YOLO_MODEL_PATH", "yolov8n.pt")
YOLO_CONFIDENCE = float(os.environ.get("YOLO_CONFIDENCE", "0.35"))
