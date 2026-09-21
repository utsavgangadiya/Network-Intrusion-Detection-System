from pathlib import Path

import joblib
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
ENCODER_PATH = BASE_DIR / "models" / "label_encoder.pkl"

# Authoritative order recorded by nids.ipynb before labels were encoded.
AUTHORITATIVE_CLASSES = [
    "Bots",
    "Brute Force",
    "DDoS",
    "DoS",
    "Normal Traffic",
    "Port Scanning",
    "Web Attacks",
]

encoder = LabelEncoder()
encoder.fit(AUTHORITATIVE_CLASSES)
assert list(encoder.classes_) == AUTHORITATIVE_CLASSES
joblib.dump(encoder, ENCODER_PATH)
print(f"Repaired {ENCODER_PATH}")
print("classes_:", list(encoder.classes_))
