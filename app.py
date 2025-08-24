from flask import Flask, render_template, Response, request, jsonify
import cv2
import os
import numpy as np
import time

from modules.facial_emotion_recognition.fer_predictor import FEPredictor
from modules.speech_emotion_recognition.ser_predictor import SERPredictor
from modules.keystroke_analysis.keystroke_predictor import KeystrokePredictor
from modules.unified_predictor.predictor import UnifiedPredictor

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# --- Model Initialization ---
# Ensure the training scripts have been run and models exist
try:
    fer_predictor = FEPredictor()
    ser_predictor = SERPredictor()
    keystroke_predictor = KeystrokePredictor()
    
    unified_predictor = UnifiedPredictor()
    unified_predictor.initialize(fer_predictor, ser_predictor, keystroke_predictor)
    print("All models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure you have run all the train_*.py scripts in the modules directory.")
    # Exit if models can't be loaded, as the app is not functional
    exit()


# Global variable to store the latest facial emotion probabilities
latest_facial_probs = {}

def generate_frames():
    """Video streaming generator function."""
    global latest_facial_probs
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            predictions, processed_frame = fer_predictor.predict(frame)
            
            # Update global facial probabilities (averaging if multiple faces)
            if predictions:
                all_probs = {label: 0.0 for label in fer_predictor.emotion_labels}
                for pred in predictions:
                    # This part needs to re-run prediction to get full probability array
                    (x,y,w,h) = pred['roi']
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    roi_gray = gray[y:y+h, x:x+w]
                    if roi_gray.size > 0:
                        roi_gray = cv2.resize(roi_gray, (48, 48))
                        roi = roi_gray.astype('float32') / 255.0
                        roi = np.expand_dims(np.expand_dims(roi, -1), 0)
                        full_preds = fer_predictor.model.predict(roi)[0]
                        for i, label in enumerate(fer_predictor.emotion_labels):
                            all_probs[label] += full_preds[i]
                
                # Average the probabilities
                num_faces = len(predictions)
                latest_facial_probs = {label: prob/num_faces for label, prob in all_probs.items()}
            else:
                latest_facial_probs = {}


            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint to process all data and return unified prediction."""
    global latest_facial_probs
    
    # 1. Process Audio Data
    audio_file = request.files.get('audio_data')
    speech_probs = {}
    if audio_file:
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"audio_{int(time.time())}.wav")
        audio_file.save(filepath)
        speech_probs = ser_predictor.predict(filepath)
        os.remove(filepath) # Clean up

    # 2. Process Keystroke Data
    keystroke_data = request.form
    keystroke_probs = {}
    if keystroke_data:
        try:
            features = np.array([
                float(keystroke_data['typingSpeed']),
                float(keystroke_data['errorRate']),
                float(keystroke_data['backspaceCount']),
                float(keystroke_data['sessionDuration'])
            ])
            keystroke_probs = keystroke_predictor.predict(features)
        except (KeyError, ValueError) as e:
            print(f"Could not parse keystroke data: {e}")

    # 3. Get Facial Data (from global var)
    facial_probs = latest_facial_probs

    # Check if we have any data at all
    if not facial_probs and not speech_probs and not keystroke_probs:
        return jsonify({
            "error": "No data received. Please enable your camera, microphone, or type some text."
        })

    # 4. Get Unified Prediction
    dominant_mood, final_probabilities, recommendation = unified_predictor.get_overall_mood(
        facial_probs, speech_probs, keystroke_probs
    )

    return jsonify({
        "dominantMood": dominant_mood,
        "probabilities": final_probabilities,
        "recommendation": recommendation,
        "sources": {
            "face": facial_probs,
            "speech": speech_probs,
            "keystroke": keystroke_probs
        }
    })

if __name__ == '__main__':
    app.run(debug=True)