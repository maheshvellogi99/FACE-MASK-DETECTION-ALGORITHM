# Face Mask Detection System

## Description
This project implements a real-time face mask detection system using computer vision and deep learning. It can detect whether a person is wearing a face mask or not through webcam feed.

## Features
- Real-time face detection and mask classification
- Voice announcements for mask/no-mask status
- Email alerts for repeated mask violations
- Visual feedback with bounding boxes and confidence scores
- Support for multiple face detection
- Easy-to-use interface with quit options

## Requirements
- Python 3.x
- OpenCV
- TensorFlow/Keras
- pyttsx3
- imutils
- numpy
- Gmail account (for email alerts)

## Installation
1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```
3. Download the face detector model files:
   - Place `deploy.prototxt` and `res10_300x300_ssd_iter_140000.caffemodel` in the `face_detector` folder
4. Download the mask detector model:
   - Place `mask_detector.keras` in the root directory

## Usage
1. Run the main script:
```bash
python mask.py
```
2. The webcam will activate and start detecting faces
3. Press 'q' or 'ESC' to quit the application

## Project Structure
```
├── mask.py              # Main application file
├── test.py             # Test version without email alerts
├── face_detector/      # Face detection model files
│   ├── deploy.prototxt
│   └── res10_300x300_ssd_iter_140000.caffemodel
├── mask_detector.keras # Mask classification model
└── violations/         # Directory for storing violation images
```

## Email Configuration
To enable email alerts, update the following variables in `mask.py`:
- `EMAIL_SENDER`: Your Gmail address
- `EMAIL_PASSWORD`: Your Gmail app password
- `EMAIL_RECEIVER`: Recipient email address

## Contributing
Feel free to submit issues and enhancement requests!

## License
This project is licensed under the MIT License - see the LICENSE file for details. 