# -*- coding: utf-8 -*-
import os
import time
import sys
import streamlit as st

# ========== HANDLE OPENCV IMPORT ==========
try:
    import cv2
    import numpy as np
    OPENCV_AVAILABLE = True
except ImportError as e:
    OPENCV_AVAILABLE = False
    st.error(f"OpenCV import error: {e}")
    st.info("Please check requirements.txt has opencv-python-headless")
    st.stop()

# ========== HANDLE MEDIAPIPE IMPORT ==========
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError as e:
    MEDIAPIPE_AVAILABLE = False
    st.error(f"MediaPipe import error: {e}")
    st.stop()

from config import PERCLOS_WINDOW, PERCLOS_THRESHOLD, YAWN_THRESHOLD, ALARM_COOLDOWN
from utils import get_eye_rois, get_mouth_roi, preprocess_roi, PERCLOSDetector, draw_futuristic_overlay, play_alarm_html
from predictor import DrowsinessPredictor

st.set_page_config(page_title="DrowsGuard", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")

def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.markdown("<style>.main-header{text-align:center;font-size:2rem;color:#00ff88}</style>", unsafe_allow_html=True)

def init_session_state():
    defaults = {
        "running": False, "predictor": None, "face_mesh": None, "detector": None,
        "total_frames": 0, "drowsy_count": 0, "session_start": time.time(),
        "last_status": "normal", "last_eye_state": "open_eye", "last_mouth_state": "open_eye",
        "last_confidence": 0.0, "last_perclos": 0.0, "last_yawn_count": 0,
        "mediapipe_error": None
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
    if not MEDIAPIPE_AVAILABLE:
        return None
    try:
        mp_face = mp.solutions.face_mesh
        return mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.session_state.mediapipe_error = str(e)
        return None

def main():
    load_css()
    init_session_state()
    
    st.markdown('<div class="main-header">🚗 DROWSGUARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
    
    if not OPENCV_AVAILABLE:
        st.error("❌ OpenCV not available. Please check requirements.txt")
        st.stop()
    
    with st.spinner("Loading AI Models..."):
        if st.session_state.predictor is None:
            st.session_state.predictor = load_predictor()
        if st.session_state.face_mesh is None:
            st.session_state.face_mesh = load_face_mesh()
        if st.session_state.detector is None:
            st.session_state.detector = PERCLOSDetector(
                window=PERCLOS_WINDOW, 
                perclos_thresh=PERCLOS_THRESHOLD, 
                yawn_thresh=YAWN_THRESHOLD, 
                alarm_cooldown=ALARM_COOLDOWN
            )
    
    model_ready = st.session_state.predictor and st.session_state.predictor.is_ready()
    facemesh_ready = st.session_state.face_mesh is not None
    
    if not model_ready:
        st.error("❌ Models failed to load. Check HuggingFace links in config.py")
        st.stop()
    
    if not facemesh_ready:
        st.error("❌ MediaPipe FaceMesh failed to load")
        st.stop()
    
    col_cam, col_info = st.columns([3, 2], gap="large")
    
    with col_cam:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📷 CAMERA FEED")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("▶ START", use_container_width=True, disabled=st.session_state.running):
                st.session_state.running = True
                st.session_state.detector.reset()
                st.session_state.session_start = time.time()
                st.session_state.total_frames = 0
                st.session_state.drowsy_count = 0
                st.rerun()
        with col2:
            if st.button("⏹ STOP", use_container_width=True, disabled=not st.session_state.running):
                st.session_state.running = False
                st.rerun()
        with col3:
            if st.button("🔄 RESET", use_container_width=True):
                st.session_state.detector.reset()
                st.session_state.total_frames = 0
                st.session_state.drowsy_count = 0
                st.session_state.session_start = time.time()
        
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        alert_placeholder = st.empty()
        
        if st.session_state.running:
            cap = None
            try:
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Cannot access camera. Please check permissions.")
                    st.session_state.running = False
                else:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    frame_idx = 0
                    
                    while st.session_state.running:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_idx += 1
                        st.session_state.total_frames += 1
                        h, w = frame.shape[:2]
                        
                        if frame_idx % 2 == 0:
                            try:
                                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                results = st.session_state.face_mesh.process(rgb)
                                
                                if results and results.multi_face_landmarks:
                                    lm = results.multi_face_landmarks[0].landmark
                                    left_eye, right_eye = get_eye_rois(frame, lm, h, w)
                                    mouth = get_mouth_roi(frame, lm, h, w)
                                    
                                    eye_state = "open_eye"
                                    if left_eye is not None:
                                        inp = preprocess_roi(left_eye)
                                        if inp is not None:
                                            pred = st.session_state.predictor.predict(inp)
                                            if pred and not pred.get("error"):
                                                eye_state = pred["class_name"]
                                    
                                    mouth_state = "open_eye"
                                    if mouth is not None:
                                        inp = preprocess_roi(mouth)
                                        if inp is not None:
                                            pred = st.session_state.predictor.predict(inp)
                                            if pred and not pred.get("error"):
                                                mouth_state = pred["class_name"]
                                    
                                    is_drowsy, perclos, yawn_count, should_alarm = st.session_state.detector.update(
                                        eye_state, mouth_state, time.time()
                                    )
                                    
                                    st.session_state.last_perclos = perclos
                                    st.session_state.last_yawn_count = yawn_count
                                    
                                    if perclos >= PERCLOS_THRESHOLD or yawn_count >= YAWN_THRESHOLD:
                                        st.session_state.last_status = "drowsy"
                                        st.session_state.drowsy_count += 1
                                    elif perclos >= 0.4:
                                        st.session_state.last_status = "warning"
                                    else:
                                        st.session_state.last_status = "normal"
                                    
                                    if should_alarm:
                                        st.components.v1.html(play_alarm_html(), height=0, width=0)
                            except Exception as e:
                                pass
                        
                        frame = draw_futuristic_overlay(
                            frame, 
                            st.session_state.last_status, 
                            st.session_state.last_perclos, 
                            st.session_state.last_yawn_count, 
                            "", "", 0
                        )
                        video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        
                        status_class = {
                            "normal": "status-normal", 
                            "warning": "status-warning", 
                            "drowsy": "status-drowsy"
                        }.get(st.session_state.last_status, "status-normal")
                        
                        status_text = {
                            "normal": "✅ NORMAL", 
                            "warning": "⚠️ WARNING", 
                            "drowsy": "🔴 DROWSY"
                        }.get(st.session_state.last_status, "UNKNOWN")
                        
                        status_placeholder.markdown(f'<div class="status-badge {status_class}">{status_text}</div>', unsafe_allow_html=True)
                        
                        if st.session_state.last_status == "drowsy":
                            alert_placeholder.markdown('<div class="alert-box">🚨 DROWSINESS DETECTED! Please take a break!</div>', unsafe_allow_html=True)
                        else:
                            alert_placeholder.empty()
                        
                        time.sleep(0.03)
                        
            except Exception as e:
                st.error(f"Camera loop error: {e}")
            finally:
                if cap is not None:
                    cap.release()
        else:
            video_placeholder.markdown('<div class="camera-placeholder">📷<br>Camera Inactive<br>Click START to begin</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col_info:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 DASHBOARD")
        
        elapsed = int(time.time() - st.session_state.session_start)
        c1, c2, c3 = st.columns(3)
        c1.markdown(f'<div class="metric-card"><div class="metric-value">{elapsed//60:02d}:{elapsed%60:02d}</div><div class="metric-label">DURATION</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.total_frames}</div><div class="metric-label">FRAMES</div></div>', unsafe_allow_html=True)
        c3.markdown(f'<div class="metric-card"><div class="metric-value">{st.session_state.drowsy_count}</div><div class="metric-label">DROWSY</div></div>', unsafe_allow_html=True)
        
        perclos = st.session_state.last_perclos * 100
        color = "#00ff88" if perclos < 50 else "#ffc800" if perclos < 70 else "#ff3232"
        st.markdown(f'''
        <div style="margin-top:1rem">
            <div>PERCLOS (Eye Closure)</div>
            <div style="font-size:2rem;color:{color}">{perclos:.1f}%</div>
            <div class="perclos-bar-bg">
                <div class="perclos-bar-fill" style="width:{perclos}%;background:{color}"></div>
            </div>
            <div style="font-size:0.7rem;color:#64748b">Threshold: {int(PERCLOS_THRESHOLD*100)}% | Yawn: {st.session_state.last_yawn_count}/{YAWN_THRESHOLD}x</div>
        </div>
        ''', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-panel" style="margin-top:1rem">
        <b>⚙️ SYSTEM</b><br>
        • Models: InceptionV3 + MobileNetV2 + ResNet50V2<br>
        • PERCLOS Window: 30 frames<br>
        • Alert: Audio + Visual<br><br>
        <b>📖 STATUS</b><br>
        🟢 Normal - Alert<br>
        🟡 Warning - Eyes closing<br>
        🔴 Drowsy - Take a break!
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()