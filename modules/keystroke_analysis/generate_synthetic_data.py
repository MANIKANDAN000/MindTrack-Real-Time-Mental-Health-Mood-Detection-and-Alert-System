import pandas as pd
import numpy as np
import os

def generate_data(records=1000):
    # ... (the function code is the same)
    data = []
    moods = ["calm", "stressed", "anxious"]
    
    for _ in range(records):
        mood = np.random.choice(moods)
        if mood == "calm":
            typing_speed = np.random.normal(loc=70, scale=10) # WPM
            error_rate = np.random.uniform(0.01, 0.05)
            backspace_count = np.random.randint(1, 5)
            session_duration = np.random.normal(loc=300, scale=50) # seconds
        elif mood == "stressed":
            typing_speed = np.random.normal(loc=90, scale=15) # Faster, more erratic
            error_rate = np.random.uniform(0.05, 0.15)
            backspace_count = np.random.randint(5, 15)
            session_duration = np.random.normal(loc=180, scale=60)
        else: # anxious
            typing_speed = np.random.normal(loc=55, scale=12) # Slower, hesitant
            error_rate = np.random.uniform(0.04, 0.12)
            backspace_count = np.random.randint(8, 20)
            session_duration = np.random.normal(loc=400, scale=80)

        data.append({
            "typing_speed": max(0, typing_speed),
            "error_rate": max(0, error_rate),
            "backspace_count": backspace_count,
            "session_duration": max(0, session_duration),
            "mood": mood
        })
        
    return pd.DataFrame(data)

if __name__ == '__main__':
    df = generate_data(2000)
    
    # --- THIS IS THE PART TO CHECK ---
    # Construct the path to MindTrack/data/keystroke/
    output_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'keystroke')
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, 'keystroke_data.csv')
    # --- END OF CHECK ---

    df.to_csv(output_path, index=False)
    print(f"Synthetic data generated and saved to {output_path}")
    print(df.head())