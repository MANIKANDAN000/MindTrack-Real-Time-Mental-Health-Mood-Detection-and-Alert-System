import cv2
import numpy as np
import tensorflow as tf
import os
import sys

def find_required_file(filename):
    """Intelligently searches for a file in common project locations."""
    script_dir = os.path.dirname(__file__)
    # Path 1: Same directory as this script
    path1 = os.path.join(script_dir, filename)
    if os.path.exists(path1):
        return path1
    # Path 2: Project root 'models' folder (if it exists)
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    path2 = os.path.join(project_root, 'models', filename)
    if os.path.exists(path2):
        return path2
    return None

class FEPredictor:
    def __init__(self):
        model_path = find_required_file('fer_model.h5')
        cascade_path = find_required_file('haarcascade_frontalface_default.xml')
        labels_path = find_required_file('fer_class_names.npy')

        if not all([model_path, cascade_path, labels_path]):
            missing = [f for f, p in [("fer_model.h5", model_path), ("haarcascade_frontalface_default.xml", cascade_path), ("fer_class_names.npy", labels_path)] if not p]
            raise FileNotFoundError(f"Could not find required files: {', '.join(missing)}")

        print(f"INFO: Loading Haar Cascade from: {cascade_path}")
        self.model = tf.keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.emotion_labels = np.load(labels_path)

    def predict(self, frame):
        """
        Analyzes a frame, draws on it, and returns both predictions and the annotated frame.
        The prediction now includes the full probability dictionary for each face.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        
        predictions = []
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            if roi_gray.size == 0: continue
            
            # --- Perform Prediction ---
            roi_gray_resized = cv2.resize(roi_gray, (48, 48))
            roi = np.expand_dims(roi_gray_resized, axis=-1)
            roi = np.expand_dims(roi, axis=0)
            roi = roi / 255.0  # Normalization
            
            # This single predict call gets all probabilities
            all_preds = self.model.predict(roi, verbose=0)[0]
            
            emotion_index = all_preds.argmax()
            emotion = self.emotion_labels[emotion_index]
            confidence = all_preds[emotion_index]
            
            # Create a full dictionary of {label: probability}
            full_probs = {label: float(prob) for label, prob in zip(self.emotion_labels, all_preds)}

            # Append all data needed by the app
            predictions.append({
                'roi': [x, y, w, h],
                'emotion': emotion,
                'confidence': float(confidence),
                'probabilities': full_probs  # <-- CRITICAL ADDITION
            })
            
            # --- Draw on the frame ---
            box_color = (0, 255, 0) if emotion not in ['angry', 'sad', 'fear'] else (0, 0, 255)
            label = f"{emotion}: {confidence:.2f}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), box_color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
            
        return predictions, frame