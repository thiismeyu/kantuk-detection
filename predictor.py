# -*- coding: utf-8 -*-
import os
import requests
import numpy as np
import tensorflow as tf
from config import MODEL_PATHS, HF_MODELS, MODELS_DIR, CLASS_NAMES, NUM_CLASSES, CONFIDENCE_MIN

os.makedirs(MODELS_DIR, exist_ok=True)

def download_models():
    for local_name, url in HF_MODELS.items():
        path = os.path.join(MODELS_DIR, local_name)
        if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
            print(f"Model {local_name} already exists")
            continue
        try:
            print(f"Downloading {local_name}...")
            r = requests.get(url, stream=True, timeout=120)
            if r.status_code == 200:
                with open(path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded {local_name}")
            else:
                print(f"Failed to download {local_name}: {r.status_code}")
        except Exception as e:
            print(f"Error downloading {local_name}: {e}")

class DrowsinessPredictor:
    def __init__(self, val_accuracies=None):
        self.models = {}
        self.weights = {}
        self.is_loaded = False
        self.load_error = None
        download_models()
        self._load_models(val_accuracies)
    
    def _load_models(self, val_accuracies):
        loaded_accs = {}
        for name, path in MODEL_PATHS.items():
            if not os.path.exists(path):
                print(f"Model {name} not found at {path}")
                continue
            try:
                model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
                self.models[name] = model
                acc = (val_accuracies or {}).get(name, 100.0)
                loaded_accs[name] = acc
                print(f"Loaded {name}")
            except Exception as e:
                print(f"Error loading {name}: {e}")
        if not self.models:
            self.load_error = "No models could be loaded"
            return
        total = sum(loaded_accs.values())
        self.weights = {name: acc/total for name, acc in loaded_accs.items()} if total > 0 else {name: 1/len(self.models) for name in self.models}
        self.is_loaded = True
    
    def predict(self, roi_input):
        if not self.is_loaded or roi_input is None:
            return {"class_name": "unknown", "confidence": 0.0, "per_model": {}, "is_reliable": False, "error": self.load_error}
        try:
            combined = np.zeros(NUM_CLASSES)
            per_model = {}
            for name, model in self.models.items():
                prob = model.predict(roi_input, verbose=0)[0]
                w = self.weights.get(name, 1.0 / len(self.models))
                combined += w * prob
                pred_idx = np.argmax(prob)
                per_model[name] = {"class": CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else "unknown", "confidence": float(prob[pred_idx])}
            final_idx = np.argmax(combined)
            final_class = CLASS_NAMES[final_idx] if final_idx < len(CLASS_NAMES) else "unknown"
            return {"class_name": final_class, "confidence": float(combined[final_idx]), "per_model": per_model, "is_reliable": combined[final_idx] >= CONFIDENCE_MIN, "error": None}
        except Exception as e:
            return {"class_name": "error", "confidence": 0.0, "per_model": {}, "is_reliable": False, "error": str(e)}
    
    def is_ready(self):
        return self.is_loaded and len(self.models) > 0