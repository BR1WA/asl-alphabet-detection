import os
import base64
import numpy as np
import cv2
import mediapipe as mp
import pickle
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)
CORS(app)

# Load models
model_mobilenet = load_model('model/asl_model.keras')
model_rf, label_encoder = joblib.load('model/asl_landmark_rf_model_enhanced.pkl')

# Define ASL letters
ASL_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# ✅ Feature Engineering for RF Model
def extract_features_from_landmarks(landmarks):
    landmarks_np = np.array(landmarks).reshape(-1, 3)
    key_pairs = [
        (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
        (4, 8), (8, 12), (12, 16), (16, 20),
        (5, 17), (2, 5), (14, 17)
    ]
    distances = []
    for i, j in key_pairs:
        dist = np.linalg.norm(landmarks_np[i] - landmarks_np[j])
        distances.append(dist)
    return np.concatenate([landmarks, distances])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        image_data = data['image']
        model_type = data.get('model_type', 'mobilenet')  # Default to mobilenet

        # Decode image
        encoded_image = image_data.split(',')[1]
        np_arr = np.frombuffer(base64.b64decode(encoded_image), np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction = 'nothing'
        confidence = 0.0
        skeleton_canvas = np.zeros((h, w, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if model_type == 'mobilenet':
                    x_coords = [lm.x for lm in hand_landmarks.landmark]
                    y_coords = [lm.y for lm in hand_landmarks.landmark]
                    x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
                    y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

                    offset = 20
                    x_min = max(0, x_min - offset)
                    y_min = max(0, y_min - offset)
                    x_max = min(w, x_max + offset)
                    y_max = min(h, y_max + offset)

                    mp_drawing.draw_landmarks(skeleton_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    skeleton_roi = skeleton_canvas[y_min:y_max, x_min:x_max]

                    if skeleton_roi.size > 0:
                        skeleton_img = cv2.resize(skeleton_roi, (224, 224))
                        skeleton_img = skeleton_img.astype("float32") / 255.0
                        skeleton_img = np.expand_dims(skeleton_img, axis=0)

                        model_prediction = model_mobilenet.predict(skeleton_img, verbose=0)
                        predicted_class_idx = np.argmax(model_prediction)
                        confidence = float(np.max(model_prediction))
                        prediction = ASL_CLASSES[predicted_class_idx]

                elif model_type == 'random_forest':
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])

                    # 🛠 Apply feature engineering (75 features)
                    features = extract_features_from_landmarks(landmarks).reshape(1, -1)

                    model_prediction = model_rf.predict_proba(features)[0]
                    predicted_class_idx = np.argmax(model_prediction)
                    confidence = float(np.max(model_prediction))
                    prediction = label_encoder.inverse_transform([predicted_class_idx])[0]

        return jsonify({"prediction": prediction, "confidence": confidence})

    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"prediction": "error", "confidence": 0.0})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
