"""
Simplified Facial Detection App (No TensorFlow Required)
Detects faces in real-time using camera feed
Compatible with Python 3.14
"""

import cv2
import numpy as np
import random

class SimpleFaceDetector:
    def __init__(self):
        self.emotions = ['Happy', 'Sad', 'Angry', 'Surprise', 'Neutral', 'Fear', 'Disgust']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        # Color scheme for emotions
        self.emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 165, 0), # Orange
            'Neutral': (128, 128, 128) # Gray
        }
    
    def detect_emotion(self, face_roi, gray_roi):
        """Simple emotion detection based on facial features"""
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 10)
        
        # Detect smile
        smiles = self.smile_cascade.detectMultiScale(gray_roi, 1.8, 20)
        
        # Simple heuristic-based emotion detection
        if len(smiles) > 0:
            return 'Happy', 0.85
        elif len(eyes) == 0:
            return 'Neutral', 0.70
        elif len(eyes) > 2:
            return 'Surprise', 0.75
        else:
            return 'Neutral', 0.65
    
    def run(self):
        """Run the face detection application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            print("\nTroubleshooting:")
            print("1. Make sure no other application is using the camera")
            print("2. Check camera permissions in Windows Settings")
            print("3. Try unplugging and replugging the camera")
            return
        
        print("Starting face detection...")
        print("Press 'q' to quit")
        print("\nNote: This is a simplified version that detects faces and smiles.")
        print("For full emotion recognition, you'll need Python 3.11 or lower.")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5, 
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                gray_roi = gray[y:y+h, x:x+w]
                
                # Detect emotion
                emotion, confidence = self.detect_emotion(face_roi, gray_roi)
                
                # Get color for this emotion
                color = self.emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                # Display emotion and confidence
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                
                # Detect and draw eyes
                eyes = self.eye_cascade.detectMultiScale(gray_roi, 1.1, 10)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(face_roi, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # Detect and draw smile
                smiles = self.smile_cascade.detectMultiScale(gray_roi, 1.8, 20)
                for (sx, sy, sw, sh) in smiles:
                    cv2.rectangle(face_roi, (sx, sy), (sx+sw, sy+sh), (0, 255, 255), 2)
            
            # Add info text
            info_text = [
                "Press 'q' to quit",
                f"Faces detected: {len(faces)}",
                "Simplified version (no TensorFlow)"
            ]
            
            y_offset = 30
            for text in info_text:
                cv2.putText(frame, text, (10, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                y_offset += 30
            
            # Display the frame
            cv2.imshow('Face Detection (Simplified)', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\nApplication closed")
        print(f"Total frames processed: {frame_count}")

def main():
    """Main function to run the face detection app"""
    print("=" * 60)
    print("SIMPLIFIED FACE DETECTION APP")
    print("=" * 60)
    print("\nThis is a simplified version that works with Python 3.14")
    print("It detects faces and smiles but doesn't use AI for emotions.")
    print("\nFor full emotion recognition with AI:")
    print("1. Install Python 3.11 from python.org")
    print("2. Use: py -3.11 -m pip install opencv-python tensorflow numpy")
    print("3. Run: py -3.11 emotion_recognition.py")
    print("=" * 60)
    print()
    
    detector = SimpleFaceDetector()
    detector.run()

if __name__ == "__main__":
    main()