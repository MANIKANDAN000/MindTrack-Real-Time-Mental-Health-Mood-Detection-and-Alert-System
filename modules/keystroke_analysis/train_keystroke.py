import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sys # <--- IMPORT THE SYS MODULE

# --- Main Script ---
# Define path based on your provided folder structure
data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'keystroke', 'keystroke_data.csv')

# Check if the data file exists
if not os.path.exists(data_path):
    print(f"Error: Data file not found at {data_path}")
    print("Please ensure 'keystroke_data.csv' exists inside the 'data/keystroke/' folder.")
    sys.exit() # <--- USE sys.exit() INSTEAD OF exit()

print(f"Loading keystroke data from: {data_path}")
df = pd.read_csv(data_path)

# Prepare the data
# X contains the features, y contains the target label ('mood')
X = df.drop('mood', axis=1)
y = df['mood']

print("\nFeatures being used:", list(X.columns))
print("Target labels:", y.unique())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForestClassifier model
print("\n--- Starting Model Training ---")
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
print("--- Model Training Finished ---")

# Evaluate the model's performance
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nKeystroke Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model using joblib
model_path = os.path.join(os.path.dirname(__file__), 'keystroke_model.joblib')
joblib.dump(model, model_path)
print(f"\nKeystroke model saved successfully to {model_path}")