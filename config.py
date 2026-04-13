# -*- coding: utf-8 -*-
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)

# ========== HUGGINGFACE (GANTI DENGAN USERNAME DAN REPO KAMU) ==========
HF_USERNAME = "ayuuuuuuu"  # ← GANTI dengan username HuggingFace kamu!
HF_REPO = "drowsiness-model"  # ← GANTI dengan nama repo kamu!
HF_BASE = f"https://huggingface.co/{HF_USERNAME}/{HF_REPO}/resolve/main"

# Nama file HARUS persis dengan yang ada di HuggingFace
HF_MODELS = {
    "InceptionV3_after_finetune.h5": f"{HF_BASE}/InceptionV3_after_finetune.h5",
    "MobileNetV2_after_finetune.h5": f"{HF_BASE}/MobileNetV2_after_finetune.h5", 
    "ResNet50V2_after_finetune.h5": f"{HF_BASE}/ResNet50V2_after_finetune.h5",
}

MODEL_PATHS = {
    "InceptionV3": os.path.join(MODELS_DIR, "InceptionV3_after_finetune.h5"),
    "MobileNetV2": os.path.join(MODELS_DIR, "MobileNetV2_after_finetune.h5"),
    "ResNet50V2": os.path.join(MODELS_DIR, "ResNet50V2_after_finetune.h5"),
}

# ========== KONFIGURASI ==========
IMG_SIZE = (96, 96)  # Sesuaikan dengan input modelmu!
NUM_CLASSES = 3
CLASS_NAMES = ["open_eye", "closed_eye", "yawn"]

CLAHE_CLIP = 2.0
CLAHE_GRID = (4, 4)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291]

ROI_PADDING_EYE = 0.25
ROI_PADDING_MOUTH = 0.35

PERCLOS_WINDOW = 30
PERCLOS_THRESHOLD = 0.70
YAWN_THRESHOLD = 2

CONFIDENCE_MIN = 0.5
ALARM_COOLDOWN = 3.0