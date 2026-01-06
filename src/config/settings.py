import os
from dotenv import load_dotenv

load_dotenv()

PORT = int(os.getenv("PORT", 8001))
MODEL_PATH = os.getenv("MODEL_PATH", "models/student_performance_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
THRESHOLD_WEAK_SUBJECT = float(os.getenv("THRESHOLD_WEAK_SUBJECT", 5.0))
API_KEY = os.getenv("API_KEY")