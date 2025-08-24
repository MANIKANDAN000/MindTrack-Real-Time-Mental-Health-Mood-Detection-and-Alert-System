import numpy as np
import librosa
import tensorflow as tf
import os

class SERPredictor:
    def __init__(self, model_path=None, labels_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), 'ser_model.h5')
        if labels_path is None:
            labels_path = os.path.join(os.path.dirname(__file__), 'ser_labels.npy')
            
        self.model = tf.keras.models.load_model(model_path)
        self.labels = np.load(labels_path, allow_pickle=True)

    def extract_features(self, file_name):
        try:
            audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
            mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
            return mfccs
        except Exception as e:
            print(f"Error processing audio file {file_name}: {e}")
            return None

    def predict(self, audio_file_path):
        features = self.extract_features(audio_file_path)
        if features is None:
            return {label: 0.0 for label in self.labels}
        
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=2)
        
        predictions = self.model.predict(features, verbose=0)[0]
        return dict(zip(self.labels, predictions))