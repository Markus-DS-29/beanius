import os
import streamlit as st
import streamlit.components.v1 as components
import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from pydub import AudioSegment
import tempfile
import shutil
import threading
from flask import Flask, request, send_from_directory

# Create a Flask app to handle file uploads
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    if request.method == 'POST':
        audio_file = request.data
        file_path = 'audio_files/recorded_speech.wav'
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(audio_file)
        return "File received"

@app.route('/audio_recorder.js')
def serve_js():
    return send_from_directory('.', 'audio_recorder.js')

# Function to start the Flask server in a separate thread
def run_flask():
    app.run(port=5000, threaded=True)

# Streamlit app code
st.title("Welcome to the Beanius, your Espresso expert.")
st.markdown("Just give me a minute, I will be right with you.")

# Embed JavaScript for recording
components.html("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Audio Recorder</title>
    <script src="/audio_recorder.js"></script>
</head>
<body>
    <button onclick="startRecording()">Start Recording</button>
    <div id="audio-container"></div>
</body>
</html>
""", height=500)

# Run the Flask server
if __name__ == "__main__":
    threading.Thread(target=run_flask).start()

# Existing code for chat and other functionality
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Speech Recorder and Transcriber
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recorded_data' not in st.session_state:
    st.session_state.recorded_data = None
if 'fs' not in st.session_state:
    st.session_state.fs = 16000  # Sample rate required for Wav2Vec2
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

# Display recording status
status = st.empty()

def start_recording():
    st.session_state.recording = True
    st.session_state.recorded_data = None
    status.text('Recording...')
    duration = 5  # seconds

    # Define a path for the file in a persistent location
    persistent_path = "audio_files/recorded_speech.wav"
    os.makedirs(os.path.dirname(persistent_path), exist_ok=True)
    
    # Record and save the audio data
    st.session_state.recorded_data = sd.rec(int(duration * st.session_state.fs), samplerate=st.session_state.fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.session_state.recording = False
    status.text('Recording finished')

    # Save the recorded data to the persistent file
    with wave.open(persistent_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits = 2 bytes
        wf.setframerate(st.session_state.fs)
        wf.writeframes(st.session_state.recorded_data.tobytes())

    st.session_state.persistent_file_path = persistent_path

# Function to stop recording
def stop_recording():
    st.session_state.recording = False

# Layout for recording buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('Start Recording'):
        start_recording()
with col2:
    if st.button('Stop Recording') and st.session_state.recording:
        stop_recording()

# Save and plot recorded data
if 'persistent_file_path' in st.session_state:
    persistent_file_path = st.session_state.persistent_file_path
    
    st.audio(persistent_file_path, format='audio/wav')
    st.success('Recording saved successfully!')

    # Plot the recorded data
    with wave.open(persistent_file_path, 'r') as wf:
        recorded_data = np.frombuffer(wf.readframes(-1), dtype=np.int16)
    
    fig, ax = plt.subplots()
    ax.plot(recorded_data)
    ax.set_title("Recorded Audio")
    st.pyplot(fig)

    # Transcribe the audio
    if st.button('Transcribe Audio'):
        def preprocess_audio(file_path, output_path, sample_rate=16000):
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(sample_rate)  # Resample to 16 kHz
            audio = audio.set_channels(1)  # Ensure mono channel
            audio.export(output_path, format="wav")

        # Load the model and processor
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
        audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")

        processed_file_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        preprocess_audio(persistent_file_path, processed_file_path)

        # Load the processed audio file
        speech, sample_rate = sf.read(processed_file_path)

        # Display some information about the audio file
        st.write(f"Sample Rate: {sample_rate} Hz")
        st.write(f"Number of Samples: {len(speech)}")

        # Normalize the audio data
        speech = (speech - np.mean(speech)) / np.std(speech)

        # Ensure the audio input has a minimum length
        min_length = 32000  # 2 seconds of audio at 16000 Hz
        if len(speech) < min_length:
            # Pad the audio to the minimum length
            pad_length = min_length - len(speech)
            speech = np.pad(speech, (0, pad_length), mode='constant')

        # Process the audio data
        input_values = audio_processor(speech, return_tensors="pt", sampling_rate=16000).input_values

        # Perform speech-to-text with the model
        with torch.no_grad():
            logits = audio_model(input_values).logits
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = audio_processor.batch_decode(predicted_ids)

        st.session_state.transcription = transcription[0]

        # Display the transcription
        st.write("Transcription:")
        st.write(st.session_state.transcription)

# Chat Input
if prompt := st.chat_input("Say something to Espresso Martin!"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chain({"question": prompt})
    msg = response['answer']

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    with st.chat_message("assistant"):
        st.markdown(msg)
