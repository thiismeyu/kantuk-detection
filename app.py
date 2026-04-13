# -*- coding: utf-8 -*-
import os
import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

from config import (
    PERCLOS_WINDOW, PERCLOS_THRESHOLD, YAWN_THRESHOLD, ALARM_COOLDOWN,
    IMG_SIZE, CLASS_NAMES
)
from predictor import DrowsinessPredictor

st.set_page_config(page_title="DrowsGuard", page_icon="🚗", layout="wide")

# ========== CSS SEDERHANA ==========
st.markdown("""
<style>
.main-header { text-align: center; font-size: 2rem; color: #00ff88; }
.status-normal { color: #00ff88; }
.status-warning { color: #ffcc00; }
.status-drowsy { color: #ff3333; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
def init_session_state():
    defaults = {
        "running": False,
        "predictor": None,
        "face_mesh": None,
        "cap": None,
        "total_frames": 0,
        "drowsy_count": 0,
        "session_start": time.time(),
        "last_status": "normal",
        "last_perclos": 0.0,
        "last_yawn_count": 0,
        "eye_buffer": [],
        "yawn_buffer": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ========== LOAD MODELS ==========
@st.cache_resource
def load_predictor():
    try:
        p = DrowsinessPredictor()
        return p if p.is_ready() else None
    except Exception as e:
        st.error(f"Predictor error: {e}")
        return None

@st.cache_resource
def load_face_mesh():
    try:
        mp_face = mp.solutions.face_mesh
        return mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.error(f"MediaPipe error: {e}")
        return None

# ========== PREPROCESSING ==========
def preprocess_roi(roi):
    try:
        roi = cv2.resize(roi, IMG_SIZE)
        roi = roi.astype(np.float32) / 255.0
        return np.expand_dims(roi, axis=0)
    except:
        return None

# ========== PERCLOS ==========
def update_perclos(eye_state, mouth_state):
    eye_val = 1 if eye_state == "closed_eye" else 0
    yawn_val = 1 if mouth_state == "yawn" else 0
    
    st.session_state.eye_buffer.append(eye_val)
    st.session_state.yawn_buffer.append(yawn_val)
    
    if len(st.session_state.eye_buffer) > PERCLOS_WINDOW:
        st.session_state.eye_buffer.pop(0)
    if len(st.session_state.yawn_buffer) > PERCLOS_WINDOW:
        st.session_state.yawn_buffer.pop(0)
    
    perclos = sum(st.session_state.eye_buffer) / max(len(st.session_state.eye_buffer), 1)
    yawn_count = sum(st.session_state.yawn_buffer)
    
    if perclos >= PERCLOS_THRESHOLD or yawn_count >= YAWN_THRESHOLD:
        status = "drowsy"
    elif perclos >= 0.4:
        status = "warning"
    else:
        status = "normal"
    
    return status, perclos, yawn_count

# ========== ALARM ==========
def play_alarm():
    alarm_html = """
    <audio autoplay>
        <source src="https://www.soundjay.com/misc/sounds/bell-ringing-05.mp3" type="audio/mpeg">
    </audio>
    """
    st.components.v1.html(alarm_html, height=0, width=0)

# ========== MAIN ==========
def main():
    st.markdown('<div class="main-header">🚗 DROWSGUARD</div>', unsafe_allow_html=True)
    st.markdown("Driver Drowsiness Detection System")
    
    # Load models
    with st.spinner("Loading AI Models..."):
        if st.session_state.predictor is None:
            st.session_state.predictor = load_predictor()
        if st.session_state.face_mesh is None:
            st.session_state.face_mesh = load_face_mesh()
    
    if not st.session_state.predictor or not st.session_state.predictor.is_ready():
        st.error("❌ Models failed to load")
        st.stop()
    
    if not st.session_state.face_mesh:
        st.error("❌ MediaPipe failed to load")
        st.stop()
    
    # Dashboard
    col1, col2, col3 = st.columns(3)
    elapsed = int(time.time() - st.session_state.session_start)
    col1.metric("Duration", f"{elapsed//60:02d}:{elapsed%60:02d}")
    col2.metric("Frames", st.session_state.total_frames)
    col3.metric("Drowsy Events", st.session_state.drowsy_count)
    
    # PERCLOS
    perclos = st.session_state.last_perclos * 100
    st.progress(min(int(perclos), 100))
    st.caption(f"PERCLOS: {perclos:.1f}% | Yawn: {st.session_state.last_yawn_count}/{YAWN_THRESHOLD}")
    
    # Status
    status_style = {
        "normal": ("🟢 Normal - Alert", "status-normal"),
        "warning": ("🟡 Warning - Eyes closing", "status-warning"),
        "drowsy": ("🔴 DROWSY - Take a break!", "status-drowsy")
    }.get(st.session_state.last_status, ("⚪ Unknown", ""))
    
    st.markdown(f'<div class="{status_style[1]}">{status_style[0]}</div>', unsafe_allow_html=True)
    
    # Controls
    col_start, col_stop = st.columns(2)
    with col_start:
        if st.button("▶ START", use_container_width=True, disabled=st.session_state.running):
            st.session_state.running = True
            st.session_state.eye_buffer = []
            st.session_state.yawn_buffer = []
            st.session_state.session_start = time.time()
            st.session_state.total_frames = 0
            st.session_state.drowsy_count = 0
            st.rerun()
    
    with col_stop:
        if st.button("⏹ STOP", use_container_width=True, disabled=not st.session_state.running):
            st.session_state.running = False
            if st.session_state.cap:
                st.session_state.cap.release()
            st.rerun()
    
    # Camera feed
    video_placeholder = st.empty()
    
    if st.session_state.running:
        if st.session_state.cap is None or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
        
        cap = st.session_state.cap
        if cap.isOpened():
            frame_placeholder = video_placeholder.empty()
            last_alarm = 0
            
            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break
                
                st.session_state.total_frames += 1
                h, w = frame.shape[:2]
                
                # Process with MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = st.session_state.face_mesh.process(rgb)
                
                eye_state = "open_eye"
                mouth_state = "open_eye"
                
                if results.multi_face_landmarks:
                    lm = results.multi_face_landmarks[0].landmark
                    
                    # Get eye region (sederhana)
                    try:
                        eye_indices = [33, 160, 158, 133, 153, 144]
                        xs = [int(lm[i].x * w) for i in eye_indices]
                        ys = [int(lm[i].y * h) for i in eye_indices]
                        x1, x2 = max(0, min(xs)-10), min(w, max(xs)+10)
                        y1, y2 = max(0, min(ys)-10), min(h, max(ys)+10)
                        
                        if x2 > x1 and y2 > y1:
                            eye_roi = frame[y1:y2, x1:x2]
                            if eye_roi.size > 0:
                                inp = preprocess_roi(eye_roi)
                                if inp is not None:
                                    pred = st.session_state.predictor.predict(inp)
                                    if pred and not pred.get("error"):
                                        eye_state = pred["class_name"]
                    except:
                        pass
                    
                    # Get mouth region
                    try:
                        mouth_indices = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291]
                        xs = [int(lm[i].x * w) for i in mouth_indices]
                        ys = [int(lm[i].y * h) for i in mouth_indices]
                        x1, x2 = max(0, min(xs)-15), min(w, max(xs)+15)
                        y1, y2 = max(0, min(ys)-10), min(h, max(ys)+10)
                        
                        if x2 > x1 and y2 > y1:
                            mouth_roi = frame[y1:y2, x1:x2]
                            if mouth_roi.size > 0:
                                inp = preprocess_roi(mouth_roi)
                                if inp is not None:
                                    pred = st.session_state.predictor.predict(inp)
                                    if pred and not pred.get("error"):
                                        mouth_state = pred["class_name"]
                    except:
                        pass
                
                # Update PERCLOS
                status, perclos_val, yawn_count = update_perclos(eye_state, mouth_state)
                st.session_state.last_status = status
                st.session_state.last_perclos = perclos_val
                st.session_state.last_yawn_count = yawn_count
                
                if status == "drowsy":
                    st.session_state.drowsy_count += 1
                    current_time = time.time()
                    if current_time - last_alarm > ALARM_COOLDOWN:
                        last_alarm = current_time
                        play_alarm()
                
                # Draw overlay
                status_text = {
                    "normal": "NORMAL",
                    "warning": "WARNING",
                    "drowsy": "DROWSY!"
                }.get(status, "UNKNOWN")
                
                color = {
                    "normal": (0, 255, 0),
                    "warning": (0, 255, 255),
                    "drowsy": (0, 0, 255)
                }.get(status, (255, 255, 255))
                
                cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.putText(frame, f"PERCLOS: {perclos_val*100:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                cv2.putText(frame, f"Yawn: {yawn_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                # Display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                time.sleep(0.03)
                st.rerun()
        else:
            st.error("Cannot access camera")
            st.session_state.running = False
    else:
        video_placeholder.info("Click START to begin detection")

if __name__ == "__main__":
    init_session_state()
    main()