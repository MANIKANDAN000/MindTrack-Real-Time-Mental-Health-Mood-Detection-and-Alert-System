class UnifiedPredictor:
    def __init__(self):
        # This will be initialized in the main app to avoid loading models multiple times
        self.fer_predictor = None
        self.ser_predictor = None
        self.keystroke_predictor = None
    
    def initialize(self, fer_predictor, ser_predictor, keystroke_predictor):
        self.fer_predictor = fer_predictor
        self.ser_predictor = ser_predictor
        self.keystroke_predictor = keystroke_predictor

    def get_overall_mood(self, facial_probs, speech_probs, keystroke_probs, weights=None):
        if weights is None:
            weights = {'face': 0.5, 'speech': 0.3, 'keystroke': 0.2}

        # Define mapping from detailed emotions to broader mental states
        # Keys are from FER, SER, and Keystroke models
        # Values are the target states: 'Positive', 'Neutral', 'Stressed', 'Anxious', 'Depressed'
        emotion_to_state_map = {
            # Facial
            'Happy': 'Positive', 'Surprise': 'Positive', 'Neutral': 'Neutral',
            'Sad': 'Depressed', 'Angry': 'Stressed', 'Fear': 'Anxious', 'Disgust': 'Stressed',
            # Speech
            'calm': 'Neutral', 'happy': 'Positive', 'sad': 'Depressed',
            'angry': 'Stressed', 'fearful': 'Anxious',
            # Keystroke
            'calm': 'Neutral', 'stressed': 'Stressed', 'anxious': 'Anxious'
        }
        
        target_states = ['Positive', 'Neutral', 'Stressed', 'Anxious', 'Depressed']
        state_scores = {state: 0.0 for state in target_states}

        # Aggregate scores from Facial Recognition
        for emotion, prob in facial_probs.items():
            state = emotion_to_state_map.get(emotion)
            if state:
                state_scores[state] += prob * weights['face']

        # Aggregate scores from Speech Recognition
        for emotion, prob in speech_probs.items():
            state = emotion_to_state_map.get(emotion)
            if state:
                state_scores[state] += prob * weights['speech']

        # Aggregate scores from Keystroke Analysis
        for mood, prob in keystroke_probs.items():
            state = emotion_to_state_map.get(mood)
            if state:
                state_scores[state] += prob * weights['keystroke']

        # Normalize the scores to get a final probability distribution
        total_score = sum(state_scores.values())
        if total_score == 0:
            return "Undetermined", {"Undetermined": 1.0}, "Could not determine mood. Please provide more data."

        final_probabilities = {state: score / total_score for state, score in state_scores.items()}
        
        # Determine the dominant state
        dominant_mood = max(final_probabilities, key=final_probabilities.get)
        
        # Get recommendation
        recommendation = self.get_recommendation(dominant_mood)

        return dominant_mood, final_probabilities, recommendation

    def get_recommendation(self, mood):
        recommendations = {
            "Positive": "You seem to be in a great mood! Keep up the positive energy. Maybe a short walk outside could make your day even better.",
            "Neutral": "You seem to be in a calm and neutral state. This is a good time for focused work or a relaxing activity like reading.",
            "Stressed": "Signs of stress detected. Consider taking a 5-minute break. Practice deep breathing: inhale for 4 seconds, hold for 4, and exhale for 6. Repeat 5 times.",
            "Anxious": "Signs of anxiety detected. Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you can touch, 3 you can hear, 2 you can smell, and 1 you can taste.",
            "Depressed": "We've noticed patterns that may indicate a low mood. It's important to be kind to yourself. Consider reaching out to a friend, family member, or a mental health professional. You are not alone."
        }
        return recommendations.get(mood, "No specific recommendation available.")