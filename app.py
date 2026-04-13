# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import streamlit as st
from PIL import Image
import io

# WebRTC untuk kamera (pengganti OpenCV)
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# MediaPipe tetap dipakai
import mediapipe as mp

from config import (
    PERCLOS_WINDOW, PERCLOS_THRESHOLD, YAWN_THRESHOLD, ALARM_COOLDOWN,
    IMG_SIZE, CLASS_NAMES
)
from predictor import DrowsinessPredictor

st.set_page_config(page_title="DrowsGuard", page_icon="🚗", layout="wide", initial_sidebar_state="collapsed")

# ========== CSS ==========
def load_css():
    css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except:
        st.markdown("<style>.main-header{text-align:center;font-size:2rem;color:#00ff88}</style>", unsafe_allow_html=True)

# ========== SESSION STATE ==========
def init_session_state():
    defaults = {
        "predictor": None,
        "face_mesh": None,
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
        return None

# ========== PREPROCESSING (Tanpa OpenCV) ==========
def preprocess_image(image_array):
    """Preprocess image without OpenCV"""
    try:
        from PIL import Image
        import numpy as np
        
        # Convert to PIL Image if needed
        if isinstance(image_array, np.ndarray):
            img = Image.fromarray(image_array)
        else:
            img = image_array
        
        # Resize
        img = img.resize(IMG_SIZE, Image.LANCZOS)
        
        # Convert to array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Add batch dimension
        return np.expand_dims(img_array, axis=0)
    except Exception as e:
        return None

# ========== PERCLOS ==========
def update_perclos(eye_state, mouth_state):
    buffer_size = PERCLOS_WINDOW
    
    eye_val = 1 if eye_state == "closed_eye" else 0
    yawn_val = 1 if mouth_state == "yawn" else 0
    
    st.session_state.eye_buffer.append(eye_val)
    st.session_state.yawn_buffer.append(yawn_val)
    
    if len(st.session_state.eye_buffer) > buffer_size:
        st.session_state.eye_buffer.pop(0)
    if len(st.session_state.yawn_buffer) > buffer_size:
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

# ========== VIDEO PROCESSOR ==========
class DrowsinessVideoProcessor(VideoProcessorBase):
    def __init__(self, face_mesh, predictor):
        self.face_mesh = face_mesh
        self.predictor = predictor
        self.last_alarm = 0
        
    def recv(self, frame):
        from streamlit_webrtc import VideoFrame
        import numpy as np
        
        img = frame.to_ndarray(format="rgb24")
        h, w = img.shape[:2]
        
        # Process with MediaPipe
        results = self.face_mesh.process(img)
        
        eye_state = "open_eye"
        mouth_state = "open_eye"
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            # Get eye region (simplified)
            try:
                # Left eye landmarks
                left_eye_pts = []
                for idx in [33, 160, 158, 133, 153, 144]:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    left_eye_pts.append([x, y])
                
                if left_eye_pts:
                    # Crop eye region
                    xs = [p[0] for p in left_eye_pts]
                    ys = [p[1] for p in left_eye_pts]
                    x1, x2 = max(0, min(xs)-10), min(w, max(xs)+10)
                    y1, y2 = max(0, min(ys)-10), min(h, max(ys)+10)
                    
                    if x2 > x1 and y2 > y1:
                        eye_roi = img[y1:y2, x1:x2]
                        if eye_roi.size > 0:
                            inp = preprocess_image(eye_roi)
                            if inp is not None:
                                pred = self.predictor.predict(inp)
                                if pred and not pred.get("error"):
                                    eye_state = pred["class_name"]
            except:
                pass
            
            # Get mouth region (simplified)
            try:
                mouth_pts = []
                for idx in [61, 185, 40, 39, 37, 267, 269, 270, 409, 291]:
                    x = int(lm[idx].x * w)
                    y = int(lm[idx].y * h)
                    mouth_pts.append([x, y])
                
                if mouth_pts:
                    xs = [p[0] for p in mouth_pts]
                    ys = [p[1] for p in mouth_pts]
                    x1, x2 = max(0, min(xs)-15), min(w, max(xs)+15)
                    y1, y2 = max(0, min(ys)-10), min(h, max(ys)+10)
                    
                    if x2 > x1 and y2 > y1:
                        mouth_roi = img[y1:y2, x1:x2]
                        if mouth_roi.size > 0:
                            inp = preprocess_image(mouth_roi)
                            if inp is not None:
                                pred = self.predictor.predict(inp)
                                if pred and not pred.get("error"):
                                    mouth_state = pred["class_name"]
            except:
                pass
        
        # Update PERCLOS
        status, perclos, yawn_count = update_perclos(eye_state, mouth_state)
        st.session_state.last_status = status
        st.session_state.last_perclos = perclos
        st.session_state.last_yawn_count = yawn_count
        
        if status == "drowsy":
            st.session_state.drowsy_count += 1
            current_time = time.time()
            if current_time - st.session_state.get("last_alarm_time", 0) > ALARM_COOLDOWN:
                st.session_state.last_alarm_time = current_time
                # Alarm akan di-trigger di main thread
        
        # Draw overlay (simple)
        status_text = {"normal": "✅ NORMAL", "warning": "⚠️ WARNING", "drowsy": "🔴 DROWSY"}.get(status, "NORMAL")
        color = {"normal": (0,255,0), "warning": (255,255,0), "drowsy": (0,0,255)}.get(status, (255,255,255))
        
        # Draw text on image
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(img_bgr, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(img_bgr, f"PERCLOS: {perclos*100:.0f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        
        return VideoFrame.from_ndarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), format="rgb24")

# ========== MAIN ==========
def main():
    load_css()
    init_session_state()
    
    st.markdown('<div class="main-header">🚗 DROWSGUARD</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Driver Drowsiness Detection System</div>', unsafe_allow_html=True)
    
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
    
    # PERCLOS display
    perclos = st.session_state.last_perclos * 100
    st.progress(min(int(perclos), 100))
    st.caption(f"PERCLOS: {perclos:.1f}% | Yawn: {st.session_state.last_yawn_count}/{YAWN_THRESHOLD}")
    
    # Status
    status_color = {
        "normal": "🟢 Normal - Alert",
        "warning": "🟡 Warning - Eyes closing",
        "drowsy": "🔴 DROWSY - Take a break!"
    }.get(st.session_state.last_status, "⚪ Unknown")
    st.markdown(f"### {status_color}")
    
    # Alarm trigger
    if st.session_state.last_status == "drowsy":
        st.markdown("""
        <div style="background:rgba(255,0,0,0.3);border:2px solid red;border-radius:10px;padding:1rem;text-align:center">
            🚨 DROWSINESS DETECTED! 🚨<br>
            Please pull over and take a break immediately!
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.get("last_alarm_time", 0) > 0:
            st.components.v1.html(play_alarm_html(), height=0, width=0)
    
    # WebRTC Camera
    webrtc_ctx = webrtc_streamer(
        key="drowsiness-detection",
        video_processor_factory=lambda: DrowsinessVideoProcessor(
            st.session_state.face_mesh, 
            st.session_state.predictor
        ),
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
    )
    
    # Update frame count
    if webrtc_ctx.state.playing:
        st.session_state.total_frames += 1
        time.sleep(0.03)
        st.rerun()

if __name__ == "__main__":
    main()