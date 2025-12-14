# ASL Real-time Recognition Project

This project integrates a pre-trained Keras MobileNet model with a web frontend to provide real-time American Sign Language (ASL) letter recognition using webcam input.

## Project Structure

```
asl_project/
├── app.py                 # Flask backend application
├── model/
│   └── asl_model.keras   # Pre-trained TensorFlow/Keras model
├── templates/
│   └── index.html        # Frontend HTML with integrated JavaScript
└── requirements.txt      # Python dependencies
```

## Features

- **Real-time ASL Detection**: Uses webcam input to recognize ASL letters A-Z
- **Interactive Frontend**: Clean, responsive web interface with practice modes
- **Flask Backend**: Serves the website and processes image predictions
- **MobileNet Model**: Lightweight CNN model optimized for real-time inference

## Installation & Setup

### Prerequisites
- Python 3.7+
- Webcam access
- Modern web browser

### Installation Steps

1. **Clone or download the project files**

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Flask application:**
   ```bash
   cd asl_project
   python app.py
   ```

4. **Access the application:**
   - Open your web browser
   - Navigate to `http://localhost:5000`
   - Allow webcam permissions when prompted

## Usage

### Freeform Practice
1. Click "Start Webcam" in the Freeform Practice section
2. Sign ASL letters A-Z in front of your webcam
3. View real-time predictions and confidence scores

### Letter Practice Mode
1. Click "Start Practicing" in the Letter Practice Mode section
2. Follow the guided practice session
3. Sign the requested letters and receive feedback

## Technical Details

### Backend (Flask)
- **Model Loading**: Loads the Keras model once at startup
- **Image Processing**: Receives base64 encoded images from frontend
- **Preprocessing**: Resizes images to 224x224, normalizes pixel values
- **Prediction**: Uses MobileNet model to classify ASL letters
- **API Endpoint**: `/predict` accepts POST requests with image data

### Frontend (JavaScript)
- **Video Capture**: Uses getUserMedia API to access webcam
- **Frame Processing**: Captures video frames to canvas, converts to base64
- **API Communication**: Sends images to Flask backend via fetch API
- **Real-time Updates**: Updates UI with predictions every 1.5 seconds

### Model Specifications
- **Input Shape**: (224, 224, 3) - RGB images
- **Output Shape**: (26,) - 26 ASL letters A-Z
- **Architecture**: MobileNet-based CNN
- **Format**: TensorFlow/Keras .keras format

## API Reference

### POST /predict
Accepts image data and returns ASL letter prediction.

**Request Body:**
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "prediction": "A",
  "confidence": 0.95
}
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**: Ensure `asl_model.keras` is in the `model/` directory
2. **Webcam Access**: Grant camera permissions in your browser
3. **Port Already in Use**: Change the port in `app.py` if 5000 is occupied
4. **Low Confidence**: Ensure good lighting and clear hand positioning

### Performance Tips

- Use good lighting for better recognition accuracy
- Position your hand clearly in the webcam view
- Sign letters distinctly and hold for a moment
- Ensure only one hand is visible in the frame

## Dependencies

See `requirements.txt` for complete list:
- Flask
- flask-cors
- tensorflow
- opencv-python
- numpy

## License

This project is for educational and demonstration purposes.

## Support

For issues or questions, please check the troubleshooting section or review the code comments for implementation details.

