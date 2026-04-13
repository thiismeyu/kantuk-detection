# fallback_face_detection.py
import cv2

class SimpleFaceDetector:
    """Fallback jika MediaPipe error - menggunakan Haar Cascade"""
    
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    def process(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            return None
        
        # Return dummy object dengan struktur mirip MediaPipe
        return DummyFaceResult(faces[0])
    
    def close(self):
        pass

class DummyFaceResult:
    def __init__(self, face_rect):
        self.multi_face_landmarks = [DummyLandmarks(face_rect)]

class DummyLandmarks:
    def __init__(self, rect):
        self.landmark = []
        x, y, w, h = rect
        # Dummy landmarks (simplifikasi)
        for i in range(468):
            self.landmark.append(DummyPoint(x + w*0.5, y + h*0.5))

class DummyPoint:
    def __init__(self, x, y):
        self.x = x
        self.y = y