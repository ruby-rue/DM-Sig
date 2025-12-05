"""
Facial Emotion Recognition using DeepFace
Real-time emotion detection with pre-trained models
"""

import cv2
from deepface import DeepFace
import time

class DeepFaceEmotionRecognizer:
    def __init__(self):
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 128, 0),    # Green
            'fear': (128, 0, 128),     # Purple
            'happy': (0, 255, 255),    # Yellow
            'sad': (255, 0, 0),        # Blue
            'surprise': (255, 165, 0), # Orange
            'neutral': (128, 128, 128) # Gray
        }
        
        # Pre-load the model (optional but faster)
        print("Loading DeepFace model...")
        try:
            DeepFace.analyze(img_path="test", actions=['emotion'], enforce_detection=False)
        except:
            pass
        print("Model loaded!")
        
    def analyze_frame(self, frame):
        """Analyze a frame for emotions using DeepFace"""
        try:
            # Analyze the frame
            result = DeepFace.analyze(
                img_path=frame,
                actions=['emotion'],
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            # Handle both single face and multiple faces
            if isinstance(result, list):
                return result
            else:
                return [result]
                
        except Exception as e:
            print(f"Analysis error: {e}")
            return []
    
    def run(self):
        """Run the emotion recognition application"""
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            print("\nTroubleshooting:")
            print("1. Make sure no other app is using the camera")
            print("2. Check camera permissions in Windows Settings")
            print("3. Try camera_test.py to find the correct camera index")
            return
        
        print("\n" + "="*60)
        print("DeepFace Emotion Recognition Started!")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("="*60 + "\n")
        
        frame_count = 0
        last_analysis_time = 0
        analysis_interval = 0.5  # Analyze every 0.5 seconds (to reduce lag)
        last_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Only analyze every few frames to improve performance
            if current_time - last_analysis_time >= analysis_interval:
                last_results = self.analyze_frame(frame)
                last_analysis_time = current_time
            
            # Draw results on frame
            for result in last_results:
                try:
                    # Get face region
                    region = result.get('region', {})
                    x = region.get('x', 0)
                    y = region.get('y', 0)
                    w = region.get('w', 0)
                    h = region.get('h', 0)
                    
                    # Get dominant emotion
                    emotions = result.get('emotion', {})
                    dominant_emotion = result.get('dominant_emotion', 'neutral')
                    confidence = emotions.get(dominant_emotion, 0)
                    
                    # Get color for this emotion
                    color = self.emotion_colors.get(dominant_emotion.lower(), (255, 255, 255))
                    
                    # Draw rectangle around face
                    if w > 0 and h > 0:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                        
                        # Display emotion and confidence
                        label = f"{dominant_emotion.capitalize()}: {confidence:.1f}%"
                        
                        # Draw background for text
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                        cv2.rectangle(frame, (x, y-35), (x + text_size[0], y), color, -1)
                        
                        # Draw text
                        cv2.putText(frame, label, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        
                        # Show all emotions (optional)
                        y_offset = y + h + 25
                        for emotion, score in sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]:
                            text = f"{emotion}: {score:.1f}%"
                            cv2.putText(frame, text, (x, y_offset),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                            y_offset += 20
                
                except Exception as e:
                    print(f"Drawing error: {e}")
                    continue
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit | 's' to save", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(frame, f"FPS: {1/(current_time - last_analysis_time + 0.001):.1f}", 
                       (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display the frame
            cv2.imshow('DeepFace Emotion Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"emotion_capture_{int(time.time())}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved as {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n" + "="*60)
        print("Application closed")
        print(f"Total frames processed: {frame_count}")
        print("="*60)

def main():
    """Main function to run the emotion recognition app"""
    recognizer = DeepFaceEmotionRecognizer()
    recognizer.run()

if __name__ == "__main__":
    main()