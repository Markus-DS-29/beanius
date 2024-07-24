import streamlit as st
import base64
import io
import soundfile as sf

st.title("Audio Recorder")

# Function to process audio data
def process_audio(audio_data):
    audio_data = audio_data.split(",")[1]
    audio_bytes = base64.b64decode(audio_data)
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    return audio_bytes

# Display audio recorder UI
st.markdown(
    """
    <script>
    let mediaRecorder;
    let audioChunks = [];

    async function startRecording() {
        audioChunks = [];
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            mediaRecorder = new MediaRecorder(stream);
            mediaRecorder.start();

            mediaRecorder.addEventListener("dataavailable", event => {
                audioChunks.push(event.data);
            });

            mediaRecorder.addEventListener("stop", () => {
                const audioBlob = new Blob(audioChunks);
                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = function() {
                    const base64data = reader.result;
                    document.getElementById('audioData').value = base64data;
                    document.getElementById('uploadButton').click();
                };
            });

            document.getElementById('stopButton').disabled = false;
            document.getElementById('startButton').disabled = true;
            document.getElementById('status').innerText = 'Recording...';
        } catch (error) {
            console.error('Error accessing microphone:', error);
        }
    }

    function stopRecording() {
        mediaRecorder.stop();
        document.getElementById('stopButton').disabled = true;
        document.getElementById('startButton').disabled = false;
        document.getElementById('status').innerText = 'Recording stopped. Processing...';
    }
    </script>
    <button id="startButton" onclick="startRecording()">Start Recording</button>
    <button id="stopButton" onclick="stopRecording()" disabled>Stop Recording</button>
    <p id="status">Press "Start Recording" to begin.</p>
    <textarea id="audioData" style="display:none;"></textarea>
    <button id="uploadButton" style="display:none;" onclick="document.getElementById('uploadForm').submit();">Upload</button>
    <form id="uploadForm" method="POST">
        <input type="hidden" name="audio_data" id="audioDataInput">
    </form>
    """,
    unsafe_allow_html=True
)

# Capture and process the audio data
audio_data = st.experimental_get_query_params().get('audio_data', [None])[0]

if audio_data:
    audio_bytes = process_audio(audio_data)
    st.audio(audio_bytes, format='audio/wav')
    st.success("Audio file uploaded and processed successfully.")
else:
    st.warning("Please record audio first.")
