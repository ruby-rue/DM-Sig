"""
Facial Emotion Recognition App
Detects faces and recognizes emotions in real-time using camera feed
"""

import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

class EmotionRecognizer:
    def __init__(self):
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = self.create_model()
        
    def create_model(self):
        """Create a CNN model for emotion recognition"""
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.25),
            
            Flatten(),
            Dense(1024, activation='relu'),
            Dropout(0.5),
            Dense(7, activation='softmax')
        ])
        
        return model
    
    def load_weights(self, weights_path):
        """Load pre-trained weights if available"""
        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print("Weights loaded successfully!")
        else:
            print(f"Warning: Weights file not found at {weights_path}")
            print("The model will run without pre-trained weights (random predictions)")
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (48, 48))
        face_img = face_img.astype('float32') / 255.0
        face_img = np.expand_dims(face_img, axis=0)
        face_img = np.expand_dims(face_img, axis=-1)
        return face_img
    
    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        preprocessed = self.preprocess_face(face_img)
        predictions = self.model.predict(preprocessed, verbose=0)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        return self.emotion_labels[emotion_idx], confidence
    
    def run(self):
        """Run the emotion recognition application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return
        
        print("Starting emotion recognition...")
        print("Press 'q' to quit")
        
        # Set color scheme
        emotion_colors = {
            'Angry': (0, 0, 255),      # Red
            'Disgust': (0, 128, 0),    # Green
            'Fear': (128, 0, 128),     # Purple
            'Happy': (0, 255, 255),    # Yellow
            'Sad': (255, 0, 0),        # Blue
            'Surprise': (255, 165, 0), # Orange
            'Neutral': (128, 128, 128) # Gray
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
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
                
                # Predict emotion
                emotion, confidence = self.predict_emotion(face_roi)
                
                # Get color for this emotion
                color = emotion_colors.get(emotion, (255, 255, 255))
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Display emotion and confidence
                label = f"{emotion}: {confidence*100:.1f}%"
                cv2.putText(frame, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('Emotion Recognition', frame)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Application closed")

def main():
    """Main function to run the emotion recognition app"""
    recognizer = EmotionRecognizer()
    
    # Optional: Load pre-trained weights
    # You can download weights from various sources like:
    # https://github.com/oarriaga/face_classification
    weights_path = 'emotion_model_weights.h5'
    recognizer.load_weights(weights_path)
    
    # Run the application
    recognizer.run()

if __name__ == "__main__":
    main()