"""
Cute Emotion Recognition Desktop App
Standalone GUI application with animated emotion displays
No web browser needed!
"""

import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import cv2
import threading
import time
from deepface import DeepFace

class CuteEmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("✨ Emotion Recognition ✨")
        self.root.geometry("1200x800")
        self.root.configure(bg='#667eea')
        
        # Variables
        self.camera = None
        self.is_running = False
        self.current_frame = None
        self.emotions = {
            'happy': 0,
            'sad': 0,
            'angry': 0,
            'surprise': 0,
            'fear': 0,
            'disgust': 0,
            'neutral': 0
        }
        
        # Emotion emojis and colors
        self.emotion_emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'surprise': '😲',
            'fear': '😨',
            'disgust': '🤢',
            'neutral': '😐'
        }
        
        self.emotion_colors = {
            'happy': '#FEC163',
            'sad': '#4A90E2',
            'angry': '#FF6B6B',
            'surprise': '#FFA500',
            'fear': '#9B59B6',
            'disgust': '#2ECC71',
            'neutral': '#95A5A6'
        }
        
        self.setup_ui()
        self.animation_cycle = 0
        self.animate()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main container
        main_frame = tk.Frame(self.root, bg='#667eea')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(
            main_frame,
            text="✨ Emotion Recognition ✨",
            font=('Arial', 32, 'bold'),
            bg='#667eea',
            fg='white'
        )
        title_label.pack(pady=(0, 10))
        
        subtitle_label = tk.Label(
            main_frame,
            text="See your feelings come to life!",
            font=('Arial', 14),
            bg='#667eea',
            fg='white'
        )
        subtitle_label.pack(pady=(0, 20))
        
        # Content frame (video + emotions side by side)
        content_frame = tk.Frame(main_frame, bg='#667eea')
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Video
        video_frame = tk.Frame(content_frame, bg='white', relief=tk.RAISED, bd=3)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Video label
        self.video_label = tk.Label(video_frame, bg='black')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Current emotion display (overlay on video)
        self.current_emotion_frame = tk.Frame(video_frame, bg='white', relief=tk.RAISED, bd=2)
        self.current_emotion_frame.place(x=20, y=20)
        
        self.current_emotion_emoji = tk.Label(
            self.current_emotion_frame,
            text='😊',
            font=('Arial', 40),
            bg='white'
        )
        self.current_emotion_emoji.pack(side=tk.LEFT, padx=10, pady=5)
        
        self.current_emotion_text = tk.Label(
            self.current_emotion_frame,
            text='Happy',
            font=('Arial', 20, 'bold'),
            bg='white',
            fg='#667eea'
        )
        self.current_emotion_text.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Controls
        controls_frame = tk.Frame(video_frame, bg='white')
        controls_frame.pack(pady=10)
        
        self.start_button = tk.Button(
            controls_frame,
            text='🎥 Start Camera',
            font=('Arial', 14, 'bold'),
            bg='#667eea',
            fg='white',
            activebackground='#764ba2',
            activeforeground='white',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            command=self.toggle_camera
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.capture_button = tk.Button(
            controls_frame,
            text='📸 Capture',
            font=('Arial', 14, 'bold'),
            bg='white',
            fg='#667eea',
            activebackground='#f0f0f0',
            activeforeground='#667eea',
            relief=tk.RAISED,
            bd=3,
            padx=20,
            pady=10,
            command=self.capture_screenshot
        )
        self.capture_button.pack(side=tk.LEFT, padx=5)
        
        # Status
        self.status_label = tk.Label(
            video_frame,
            text='Ready to start',
            font=('Arial', 12),
            bg='white',
            fg='#667eea'
        )
        self.status_label.pack(pady=5)
        
        # Right side - Emotions panel
        emotions_frame = tk.Frame(content_frame, bg='white', relief=tk.RAISED, bd=3, width=400)
        emotions_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))
        emotions_frame.pack_propagate(False)
        
        # Panel title
        panel_title = tk.Label(
            emotions_frame,
            text='Emotion Meter 💖',
            font=('Arial', 24, 'bold'),
            bg='white',
            fg='#667eea'
        )
        panel_title.pack(pady=20)
        
        # Emotion cards container
        cards_container = tk.Frame(emotions_frame, bg='white')
        cards_container.pack(fill=tk.BOTH, expand=True, padx=15)
        
        # Create emotion cards
        self.emotion_widgets = {}
        emotions_list = ['happy', 'sad', 'angry', 'surprise', 'fear', 'disgust', 'neutral']
        
        for emotion in emotions_list:
            card = self.create_emotion_card(cards_container, emotion)
            card.pack(fill=tk.X, pady=5)
            self.emotion_widgets[emotion] = card
    
    def create_emotion_card(self, parent, emotion):
        """Create a cute emotion card"""
        # Card frame
        card_frame = tk.Frame(parent, bg='#f0f0f0', relief=tk.RAISED, bd=2)
        
        # Emoji
        emoji_label = tk.Label(
            card_frame,
            text=self.emotion_emojis[emotion],
            font=('Arial', 36),
            bg='#f0f0f0'
        )
        emoji_label.pack(side=tk.LEFT, padx=15, pady=10)
        
        # Info section
        info_frame = tk.Frame(card_frame, bg='#f0f0f0')
        info_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Emotion name
        name_label = tk.Label(
            info_frame,
            text=emotion.capitalize(),
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#333',
            anchor='w'
        )
        name_label.pack(fill=tk.X)
        
        # Progress bar frame
        progress_frame = tk.Frame(info_frame, bg='white', relief=tk.SUNKEN, bd=1, height=15)
        progress_frame.pack(fill=tk.X, pady=(5, 0))
        progress_frame.pack_propagate(False)
        
        # Progress fill
        progress_fill = tk.Frame(progress_frame, bg=self.emotion_colors[emotion])
        progress_fill.place(x=0, y=0, relheight=1, width=0)
        
        # Percentage
        percentage_label = tk.Label(
            card_frame,
            text='0%',
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg=self.emotion_colors[emotion],
            width=6
        )
        percentage_label.pack(side=tk.RIGHT, padx=15)
        
        # Store references
        card_frame.emoji_label = emoji_label
        card_frame.name_label = name_label
        card_frame.progress_fill = progress_fill
        card_frame.percentage_label = percentage_label
        card_frame.emotion = emotion
        
        return card_frame
    
    def toggle_camera(self):
        """Start or stop camera"""
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start camera capture"""
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            self.status_label.config(text='❌ Camera not found')
            return
        
        self.is_running = True
        self.start_button.config(text='🛑 Stop Camera', bg='#FF6B6B')
        self.status_label.config(text='✅ Camera active - Analyzing...')
        
        # Start video thread
        video_thread = threading.Thread(target=self.update_video, daemon=True)
        video_thread.start()
        
        # Start emotion analysis thread
        analysis_thread = threading.Thread(target=self.analyze_emotions, daemon=True)
        analysis_thread.start()
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.start_button.config(text='🎥 Start Camera', bg='#667eea')
        self.status_label.config(text='Camera stopped')
        self.video_label.config(image='', bg='black')
    
    def update_video(self):
        """Update video feed"""
        while self.is_running and self.camera:
            ret, frame = self.camera.read()
            if not ret:
                break
            
            self.current_frame = frame.copy()
            
            # Convert for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit window
            img.thumbnail((640, 480), Image.Resampling.LANCZOS)
            
            photo = ImageTk.PhotoImage(image=img)
            self.video_label.config(image=photo)
            self.video_label.image = photo
            
            time.sleep(0.03)  # ~30 FPS
    
    def analyze_emotions(self):
        """Analyze emotions from video"""
        while self.is_running:
            if self.current_frame is not None:
                try:
                    result = DeepFace.analyze(
                        img_path=self.current_frame,
                        actions=['emotion'],
                        enforce_detection=False,
                        detector_backend='opencv'
                    )
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    emotions = result.get('emotion', {})
                    
                    # Normalize emotions
                    self.emotions = {
                        'happy': emotions.get('happy', 0),
                        'sad': emotions.get('sad', 0),
                        'angry': emotions.get('angry', 0),
                        'surprise': emotions.get('surprise', 0),
                        'fear': emotions.get('fear', 0),
                        'disgust': emotions.get('disgust', 0),
                        'neutral': emotions.get('neutral', 0)
                    }
                    
                    # Update UI
                    self.root.after(0, self.update_emotion_display)
                    
                except Exception as e:
                    print(f"Analysis error: {e}")
            
            time.sleep(1)  # Analyze every second
    
    def update_emotion_display(self):
        """Update emotion cards with current values"""
        # Find dominant emotion
        dominant_emotion = max(self.emotions, key=self.emotions.get)
        dominant_value = self.emotions[dominant_emotion]
        
        # Update current emotion display
        self.current_emotion_emoji.config(text=self.emotion_emojis[dominant_emotion])
        self.current_emotion_text.config(text=dominant_emotion.capitalize())
        
        # Update all emotion cards
        for emotion, value in self.emotions.items():
            card = self.emotion_widgets[emotion]
            
            # Update percentage
            card.percentage_label.config(text=f'{int(value)}%')
            
            # Update progress bar
            card.progress_fill.place(relwidth=value/100)
            
            # Highlight dominant emotion
            if emotion == dominant_emotion and value > 0:
                card.config(bg=self.emotion_colors[emotion], relief=tk.RAISED, bd=4)
                card.name_label.config(bg=self.emotion_colors[emotion], fg='white')
                card.percentage_label.config(bg=self.emotion_colors[emotion], fg='white')
                card.emoji_label.config(bg=self.emotion_colors[emotion])
            else:
                card.config(bg='#f0f0f0', relief=tk.RAISED, bd=2)
                card.name_label.config(bg='#f0f0f0', fg='#333')
                card.percentage_label.config(bg='#f0f0f0', fg=self.emotion_colors[emotion])
                card.emoji_label.config(bg='#f0f0f0')
    
    def capture_screenshot(self):
        """Save current frame"""
        if self.current_frame is not None:
            filename = f'emotion_capture_{int(time.time())}.jpg'
            cv2.imwrite(filename, self.current_frame)
            self.status_label.config(text=f'✅ Saved: {filename}')
            
            # Reset status after 2 seconds
            self.root.after(2000, lambda: self.status_label.config(
                text='✅ Camera active - Analyzing...' if self.is_running else 'Ready to start'
            ))
    
    def animate(self):
        """Add subtle animations"""
        self.animation_cycle += 1
        
        # Pulse current emotion display
        if self.animation_cycle % 20 < 10:
            self.current_emotion_frame.config(relief=tk.RAISED, bd=3)
        else:
            self.current_emotion_frame.config(relief=tk.RAISED, bd=2)
        
        # Continue animation
        self.root.after(100, self.animate)
    
    def on_closing(self):
        """Cleanup when closing"""
        self.is_running = False
        if self.camera:
            self.camera.release()
        self.root.destroy()

def main():
    """Main function"""
    print("\n" + "="*60)
    print("✨ Cute Emotion Recognition Desktop App")
    print("="*60)
    print("\nLoading DeepFace model...")
    
    # Pre-load model
    try:
        DeepFace.analyze(img_path="test", actions=['emotion'], enforce_detection=False)
    except:
        pass
    
    print("Model loaded! Starting app...")
    print("="*60 + "\n")
    
    root = tk.Tk()
    app = CuteEmotionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()