import os
import glob
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import sys

# --- Configuration ---
# This is the map the code uses. It expects a code like "01", "02", etc.
emotion_map = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
    "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
}
# These are the emotions we want to train the model on.
observed_emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "surprised"]
TARGET_FOLDER_NAME = "Audio_Song_Actors_01-24"

# --- Main Script Execution ---
print("--- MindTrack: Speech Emotion Recognition [DIAGNOSTIC MODE] ---")

# --- Step 1: Find Audio Files ---
script_path = os.path.abspath(__file__)
module_root = os.path.dirname(script_path)
project_root = os.path.abspath(os.path.join(module_root, '..', '..'))
audio_data_path = os.path.join(project_root, 'data', TARGET_FOLDER_NAME)

if not os.path.isdir(audio_data_path):
    print(f"\nFATAL ERROR: The target audio folder '{TARGET_FOLDER_NAME}' was not found at: {audio_data_path}")
    sys.exit()

search_pattern = os.path.join(audio_data_path, "Actor_*", "*.wav")
audio_files = glob.glob(search_pattern)

if not audio_files:
    print("\nFATAL ERROR: Found 0 .wav files. Please check that the 'Actor_*' folders contain audio files.")
    sys.exit()

print(f"\n[Step 1] Found {len(audio_files)} total .wav files. [OK]")

# --- Step 2: DIAGNOSE THE FIRST 10 FILENAMES ---
print("\n[Step 2] Diagnosing the first 10 filenames to check the format...")
features, labels = [], []
kept_count, skipped_count = 0, 0

for i, file_path in enumerate(audio_files):
    file_name = os.path.basename(file_path)
    
    # --- THIS IS THE DIAGNOSTIC BLOCK ---
    if i < 10: # Only print for the first 10 files to avoid flooding the console
        print("-" * 50)
        print(f"DEBUG: Processing filename: '{file_name}'")
        try:
            parts = file_name.split("-")
            print(f"DEBUG: Split filename into {len(parts)} parts: {parts}")
            
            if len(parts) > 2:
                emotion_code = parts[2]
                print(f"DEBUG: Extracted emotion code (3rd part): '{emotion_code}'")
                
                mapped_emotion = emotion_map.get(emotion_code)
                print(f"DEBUG: Looked up code in emotion_map. Result: {mapped_emotion}")
                
                if mapped_emotion in observed_emotions:
                    print("DEBUG: RESULT -> This file would be KEPT.")
                else:
                    print("DEBUG: RESULT -> This file would be SKIPPED.")
            else:
                print("DEBUG: Could not extract emotion code. Not enough parts after splitting by '-'.")
                print("DEBUG: RESULT -> This file would be SKIPPED.")

        except Exception as e:
            print(f"DEBUG: An error occurred processing this filename: {e}")
            print("DEBUG: RESULT -> This file would be SKIPPED.")

    # This is the actual processing logic from before
    try:
        parts = file_name.split("-")
        if len(parts) > 2:
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code)
            if emotion in observed_emotions:
                feature = librosa.feature.mfcc(y=librosa.load(file_path, sr=22050)[0], sr=22050, n_mfcc=40).T
                features.append(np.mean(feature, axis=0))
                labels.append(emotion)
                kept_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1
    except:
        skipped_count += 1


print("-" * 50)
print("\n--- Diagnostic Summary ---")
print(f"  - Total files processed: {len(audio_files)}")
print(f"  - Files Kept: {kept_count}")
print(f"  - Files Skipped: {skipped_count}")

# Final check
if kept_count == 0:
    print("\nFATAL ERROR: All files were skipped.")
    print("Please look at the 'DEBUG' output above. Compare your filename format to what the code expects.")
    print("The code expects a format like '03-01-08-02-02-01-12.wav', where the 3rd part ('08') is the emotion code.")
    sys.exit()

# If we get here, it means we have data! The rest of the code will now run.
# ... (The rest of the training code is the same)
print("\n[Step 3] Data loaded successfully. Preparing data for the model...")
X = np.array(features)
y = np.array(labels)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = tf.keras.utils.to_categorical(y_encoded, num_classes=len(le.classes_))
print(f"  - Data shapes: X={X.shape}, y={y_categorical.shape}")
print(f"  - Found classes: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

print("\n[Step 4] Building and training the LSTM model...")
model = Sequential([
    LSTM(128, input_shape=(X_train.shape[1], 1), return_sequences=True), Dropout(0.4),
    LSTM(64), Dropout(0.4),
    Dense(32, activation='relu'), Dropout(0.4),
    Dense(len(le.classes_), activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), metrics=['accuracy'])
model.summary()
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

print("\n[Step 5] Saving model and labels...")
model_path = os.path.join(module_root, 'ser_model.h5')
labels_path = os.path.join(module_root, 'ser_labels.npy')
model.save(model_path)
np.save(labels_path, le.classes_)
print(f"  - Model saved to: {model_path}")
print(f"  - Labels saved to: {labels_path}")
print("\n--- Training complete. ---")