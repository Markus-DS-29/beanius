import streamlit as st
import streamlit.components.v1 as components
from flask import Flask, request, jsonify
from werkzeug.wsgi import DispatcherMiddleware
from werkzeug.serving import run_simple
import base64
import io
import soundfile as sf

# Flask app for handling audio uploads
app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_audio():
    data = request.get_json()
    audio_data = data['data'].split(",")[1]
    audio_bytes = base64.b64decode(audio_data)
    
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    
    # Process the audio data (e.g., transcription)
    # Here you can save the audio file or send it to a transcription service
    
    return jsonify(status="success")

# Function to start the Flask server
def run():
    app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
        '/streamlit': st.server.app,
    })
    run_simple('localhost', 8501, app)

if __name__ == "__main__":
    run()

# Streamlit app for recording audio
def audio_recorder():
    st.markdown(
        """
        <script>
        let mediaRecorder;
        let audioChunks = [];

        async function startRecording() {
            audioChunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks);
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();

                // Send the audio blob to the backend
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = function() {
                    const base64data = reader.result;
                    fetch("/upload", {
                        method: "POST",
                        body: JSON.stringify({ data: base64data }),
                        headers: {
                            "Content-Type": "application/json"
                        }
                    });
                };
            });
        }

        function stopRecording() {
            mediaRecorder.stop();
        }

        </script>
        <button onclick="startRecording()">Start Recording</button>
        <button onclick="stopRecording()">Stop Recording</button>
        """,
        unsafe_allow_html=True
    )

audio_recorder()
