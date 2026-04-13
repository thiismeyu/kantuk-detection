# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import deque
from config import (
    IMG_SIZE, CLAHE_CLIP, CLAHE_GRID,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX,
    ROI_PADDING_EYE, ROI_PADDING_MOUTH,
    PERCLOS_WINDOW, PERCLOS_THRESHOLD, YAWN_THRESHOLD
)

# CLAHE dengan error handling
try:
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID)
except:
    # Fallback jika CLAHE gagal
    clahe = None

def crop_roi(frame, landmarks, indices, padding, h, w):
    try:
        pts = np.array([[int(landmarks[i].x * w), int(landmarks[i].y * h)] for i in indices if i < len(landmarks)])
        if len(pts) < 3:
            return None
        x1 = max(0, pts[:, 0].min() - int(padding * (pts[:, 0].max() - pts[:, 0].min())))
        y1 = max(0, pts[:, 1].min() - int(padding * (pts[:, 1].max() - pts[:, 1].min())))
        x2 = min(w, pts[:, 0].max() + int(padding * (pts[:, 0].max() - pts[:, 0].min())))
        y2 = min(h, pts[:, 1].max() + int(padding * (pts[:, 1].max() - pts[:, 1].min())))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]
    except:
        return None

def preprocess_roi(roi):
    try:
        if roi is None or roi.size == 0:
            return None
        
        # Gunakan CLAHE jika tersedia, jika tidak langsung resize
        if clahe is not None:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray = clahe.apply(gray)
            roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        roi = cv2.resize(roi, IMG_SIZE)
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)
    except:
        return None

def get_eye_rois(frame, landmarks, h, w):
    left = crop_roi(frame, landmarks, LEFT_EYE_IDX, ROI_PADDING_EYE, h, w)
    right = crop_roi(frame, landmarks, RIGHT_EYE_IDX, ROI_PADDING_EYE, h, w)
    return left, right

def get_mouth_roi(frame, landmarks, h, w):
    return crop_roi(frame, landmarks, MOUTH_IDX, ROI_PADDING_MOUTH, h, w)

class PERCLOSDetector:
    def __init__(self, window=PERCLOS_WINDOW, perclos_thresh=PERCLOS_THRESHOLD, 
                 yawn_thresh=YAWN_THRESHOLD, alarm_cooldown=3.0):
        self.window = window
        self.perclos_thresh = perclos_thresh
        self.yawn_thresh = yawn_thresh
        self.alarm_cooldown = alarm_cooldown
        self.eye_buffer = deque(maxlen=window)
        self.yawn_buffer = deque(maxlen=window)
        self.last_alarm_time = 0
        
    def update(self, eye_state, mouth_state, current_time):
        is_closed = 1 if eye_state == "closed_eye" else 0
        is_yawn = 1 if mouth_state == "yawn" else 0
        self.eye_buffer.append(is_closed)
        self.yawn_buffer.append(is_yawn)
        perclos = sum(self.eye_buffer) / max(len(self.eye_buffer), 1)
        yawn_count = sum(self.yawn_buffer)
        is_drowsy = perclos >= self.perclos_thresh or yawn_count >= self.yawn_thresh
        should_alarm = False
        if is_drowsy and (current_time - self.last_alarm_time) >= self.alarm_cooldown:
            should_alarm = True
            self.last_alarm_time = current_time
        return is_drowsy, perclos, yawn_count, should_alarm
    
    def reset(self):
        self.eye_buffer.clear()
        self.yawn_buffer.clear()
        self.last_alarm_time = 0

def draw_futuristic_overlay(frame, status, perclos, yawn_count, eye_state, mouth_state, confidence):
    h, w = frame.shape[:2]
    colors = {"normal": (0, 255, 100), "warning": (255, 200, 0), "drowsy": (255, 50, 50)}
    color = colors.get(status, (100, 100, 100))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (10, 10, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    status_text = {"normal": "✅ NORMAL", "warning": "⚠️ WARNING", "drowsy": "🔴 DROWSY"}.get(status, "UNKNOWN")
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    bar_width = int(perclos * (w - 20))
    cv2.rectangle(frame, (10, 45), (w - 10, 60), (30, 30, 40), -1)
    cv2.rectangle(frame, (10, 45), (10 + bar_width, 60), color, -1)
    cv2.putText(frame, f"PERCLOS: {perclos*100:.0f}%", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame

def play_alarm_html():
    return """
    <div style="display:none">
        <script>
        (function() {
            try {
                var audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                var now = audioCtx.currentTime;
                for (var i = 0; i < 3; i++) {
                    var osc = audioCtx.createOscillator();
                    var gain = audioCtx.createGain();
                    osc.connect(gain);
                    gain.connect(audioCtx.destination);
                    osc.frequency.value = 880;
                    gain.gain.value = 0.3;
                    osc.start(now + i * 0.5);
                    gain.gain.exponentialRampToValueAtTime(0.00001, now + i * 0.5 + 0.4);
                    osc.stop(now + i * 0.5 + 0.4);
                }
            } catch(e) {}
        })();
        </script>
    </div>
    """