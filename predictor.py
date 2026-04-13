import os
import numpy as np
import tensorflow as tf
from huggingface_hub import hf_hub_download
from config import CLASS_NAMES, NUM_CLASSES, CONFIDENCE_MIN

class DrowsinessPredictor:
    def __init__(self, val_accuracies=None):
        self.models = {}
        self.weights = {}
        self.is_loaded = False
        self.load_error = None
        self._download_and_load_models(val_accuracies)

    def _download_and_load_models(self, val_accuracies):
        print("[INFO] Menghubungkan ke Hugging Face Hub...")
        loaded_accs = {}
        
        # Ganti dengan username dan repo model kamu!
        model_repo_id = "ayuuuuuuu/drowsiness-model"
        
        model_files = {
            "InceptionV3": "InceptionV3_after_finetune.h5",
            "MobileNetV2": "MobileNetV2_after_finetune.h5",
            "ResNet50V2": "ResNet50V2_after_finetune.h5",
        }

        for name, filename in model_files.items():
            try:
                print(f"Mengunduh {filename}...")
                model_path = hf_hub_download(
                    repo_id=model_repo_id,
                    filename=filename,
                    token=os.environ.get("HF_TOKEN")
                )
                
                print(f"Memuat model {name}...")
                model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                self.models[name] = model
                
                acc = (val_accuracies or {}).get(name, 100.0)
                loaded_accs[name] = acc
                print(f"✅ {name} berhasil!")

            except Exception as e:
                print(f"Gagal memuat {name}: {e}")

        if not self.models:
            self.load_error = "Tidak ada model yang bisa dimuat."
            return

        total = sum(loaded_accs.values())
        self.weights = {name: acc/total for name, acc in loaded_accs.items()} if total > 0 else {name: 1/len(self.models) for name in self.models}
        self.is_loaded = True
        print("Semua model siap!")

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