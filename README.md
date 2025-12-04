# DM-Sig
datamining and sig project
# Facial Emotion Recognition App

A real-time facial emotion recognition application that uses your camera to detect faces and identify emotions.

## Features

- **Real-time Detection**: Detects faces in live camera feed
- **7 Emotions**: Recognizes Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral
- **Visual Feedback**: Color-coded rectangles and confidence scores
- **Easy to Use**: Simple Python implementation using OpenCV and TensorFlow

## Detected Emotions

The app can recognize 7 different emotions:
- 😠 Angry (Red)
- 🤢 Disgust (Green)
- 😨 Fear (Purple)
- 😊 Happy (Yellow)
- 😢 Sad (Blue)
- 😲 Surprise (Orange)
- 😐 Neutral (Gray)

## Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam
- pip (Python package installer)

### Setup Steps

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install opencv-python tensorflow numpy
   ```

2. **Optional - Download Pre-trained Weights:**
   
   For better accuracy, download pre-trained weights:
   - Download from: https://github.com/oarriaga/face_classification
   - Save as `emotion_model_weights.h5` in the same directory
   
   Note: The app will work without weights but predictions will be random until trained.

## Usage

Run the application:
```bash
python emotion_recognition.py
```

### Controls

- **q**: Quit the application
- The app will automatically detect faces and display emotions

### Tips for Best Results

1. Ensure good lighting
2. Face the camera directly
3. Keep your face within the camera frame
4. Maintain a distance of 2-3 feet from the camera

## How It Works

1. **Face Detection**: Uses Haar Cascade classifier to detect faces in each frame
2. **Preprocessing**: Converts detected faces to 48x48 grayscale images
3. **Emotion Prediction**: CNN model predicts emotion from preprocessed image
4. **Visualization**: Draws colored rectangles and labels on detected faces

## Model Architecture

The CNN model consists of:
- 4 Convolutional layers with ReLU activation
- 3 MaxPooling layers
- 2 Dropout layers for regularization
- 2 Dense layers
- Softmax output for 7 emotion classes

## Training Your Own Model

To train the model with your own data:

1. Prepare a dataset with labeled emotion images (48x48 grayscale)
2. Organize data into folders by emotion label
3. Add training code to the script
4. Train and save weights as `emotion_model_weights.h5`

Popular datasets for training:
- FER-2013 (Facial Expression Recognition)
- CK+ (Extended Cohn-Kanade)
- JAFFE (Japanese Female Facial Expression)

## Troubleshooting

**Camera not opening:**
- Check if another application is using the camera
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for external cameras

**Low accuracy:**
- Download and use pre-trained weights
- Ensure good lighting conditions
- Face the camera directly

**Import errors:**
- Reinstall packages: `pip install --upgrade opencv-python tensorflow`
- Check Python version compatibility

## Requirements

- Python 3.8+
- opencv-python 4.8+
- tensorflow 2.15+
- numpy 1.24+

## License

This project is for educational purposes. Feel free to modify and use as needed.

## Credits

- OpenCV for computer vision functionality
- TensorFlow/Keras for deep learning framework
- Haar Cascade classifiers from OpenCV
