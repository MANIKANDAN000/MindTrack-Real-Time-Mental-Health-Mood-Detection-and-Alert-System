document.addEventListener('DOMContentLoaded', () => {
    const recordBtn = document.getElementById('record-btn');
    const micStatus = document.getElementById('mic-status');
    const analyzeBtn = document.getElementById('analyze-btn');
    const textInput = document.getElementById('text-input');
    
    // Keystroke analysis variables
    let keyPressCount = 0;
    let backspaceCount = 0;
    let errorCount = 0; // Simplified: any non-char key could be an "error"
    let startTime = null;
    let sessionDuration = 0;
    
    // Audio recording variables
    let mediaRecorder;
    let audioChunks = [];
    let audioBlob = null;

    // --- Keystroke Logic ---
    textInput.addEventListener('focus', () => {
        if (!startTime) {
            startTime = new Date();
        }
    });

    textInput.addEventListener('input', () => {
        keyPressCount++;
    });

    textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Backspace') {
            backspaceCount++;
        }
        // Simplified error tracking
        if (!e.key.match(/^[a-zA-Z0-9\s]$/)) {
            errorCount++;
        }
    });

    // --- Audio Recording Logic ---
    recordBtn.addEventListener('click', async () => {
        if (mediaRecorder && mediaRecorder.state === 'recording') {
            mediaRecorder.stop();
            recordBtn.textContent = 'Start Recording Voice (5s)';
            recordBtn.disabled = true;
            micStatus.textContent = 'Status: Processing...';
        } else {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];

                mediaRecorder.ondataavailable = event => {
                    audioChunks.push(event.data);
                };

                mediaRecorder.onstop = () => {
                    audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    micStatus.textContent = 'Status: Recording finished. Ready for analysis.';
                    recordBtn.textContent = 'Re-record Voice (5s)';
                    recordBtn.disabled = false;
                };

                mediaRecorder.start();
                micStatus.textContent = 'Status: Recording...';
                recordBtn.textContent = 'Stop Recording';
                recordBtn.disabled = false;

                // Stop recording after 5 seconds
                setTimeout(() => {
                    if (mediaRecorder.state === 'recording') {
                        mediaRecorder.stop();
                    }
                }, 5000);

            } catch (err) {
                console.error('Error accessing microphone:', err);
                micStatus.textContent = 'Error: Could not access microphone.';
            }
        }
    });

    // --- Analysis Logic ---
    analyzeBtn.addEventListener('click', async () => {
        analyzeBtn.textContent = 'Analyzing...';
        analyzeBtn.disabled = true;

        // Calculate keystroke metrics
        if (startTime) {
            sessionDuration = (new Date() - startTime) / 1000; // in seconds
        }
        const wordsTyped = textInput.value.trim().split(/\s+/).length;
        const minutes = sessionDuration / 60;
        const typingSpeed = minutes > 0 ? Math.round(wordsTyped / minutes) : 0;
        const errorRate = keyPressCount > 0 ? errorCount / keyPressCount : 0;

        // Prepare form data
        const formData = new FormData();
        if (audioBlob) {
            formData.append('audio_data', audioBlob, 'recording.wav');
        }
        formData.append('typingSpeed', typingSpeed);
        formData.append('errorRate', errorRate);
        formData.append('backspaceCount', backspaceCount);
        formData.append('sessionDuration', sessionDuration);
        
        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });
            const result = await response.json();
            displayResults(result);
        } catch (error) {
            console.error('Analysis error:', error);
            document.getElementById('results-container').innerHTML = '<p>An error occurred during analysis. Please try again.</p>';
        } finally {
            analyzeBtn.textContent = 'Analyze My Mood';
            analyzeBtn.disabled = false;
        }
    });

    function displayResults(result) {
        if (result.error) {
            document.getElementById('results-container').innerHTML = `<p><strong>Error:</strong> ${result.error}</p>`;
            return;
        }

        const modal = document.getElementById('recommendation-modal');
        document.getElementById('modal-title').textContent = `Dominant Mood: ${result.dominantMood}`;
        document.getElementById('modal-recommendation').textContent = result.recommendation;
        
        const detailsContainer = document.getElementById('modal-details');
        detailsContainer.innerHTML = '<h4>Probability Breakdown</h4>';
        
        const probList = document.createElement('ul');
        for (const [mood, prob] of Object.entries(result.probabilities)) {
            const listItem = document.createElement('li');
            listItem.textContent = `${mood}: ${Math.round(prob * 100)}%`;
            probList.appendChild(listItem);
        }
        detailsContainer.appendChild(probList);

        modal.style.display = 'block';

        // Also update the main page results container
        document.getElementById('results-container').innerHTML = `
            <h3>Analysis Complete</h3>
            <p><strong>Overall Mood:</strong> ${result.dominantMood}</p>
            <p>A detailed recommendation has been shown.</p>
        `;
    }

    // Modal close functionality
    const modal = document.getElementById('recommendation-modal');
    const closeBtn = document.querySelector('.close-btn');
    closeBtn.onclick = () => { modal.style.display = "none"; };
    window.onclick = (event) => {
        if (event.target == modal) {
            modal.style.display = "none";
        }
    };
});