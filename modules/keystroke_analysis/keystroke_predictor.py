import joblib
import os
import numpy as np

class KeystrokePredictor:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'keystroke_model.joblib')
        
        self.model = joblib.load(model_path)
        self.labels = self.model.classes_

    def predict(self, features):
        """
        Predicts mood based on keystroke features.
        :param features: A numpy array of shape (1, 4) with [typing_speed, error_rate, backspace_count, session_duration]
        :return: A dictionary of label probabilities.
        """
        try:
            # Ensure features are in a 2D array for prediction
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
            
            probabilities = self.model.predict_proba(features)[0]
            return dict(zip(self.labels, probabilities))
        except Exception as e:
            print(f"Error during keystroke prediction: {e}")
            # Return a default neutral state in case of error
            return {label: 1.0/len(self.labels) for label in self.labels}