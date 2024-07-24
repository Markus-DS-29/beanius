import streamlit as st
import base64
import soundfile as sf
import io

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
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.start();
        mediaRecorder.addEventListener("dataavailable", event => {
            audioChunks.push(event.data);
        });
        document.getElementById('stopButton').disabled = false;
        document.getElementById('startButton').disabled = true;
        document.getElementById('status').innerText = 'Recording...';
    }

    function stopRecording() {
        mediaRecorder.stop();
        mediaRecorder.addEventListener("stop", () => {
            const audioBlob = new Blob(audioChunks);
            const reader = new FileReader();
            reader.readAsDataURL(audioBlob);
            reader.onloadend = () => {
                const base64data = reader.result;
                document.getElementById('audioData').value = base64data;
                document.getElementById('uploadAudio').click();
            };
        });
        document.getElementById('stopButton').disabled = true;
        document.getElementById('startButton').disabled = false;
        document.getElementById('status').innerText = 'Recording stopped. Processing...';
    }
    </script>
    <button id="startButton" onclick="startRecording()">Start Recording</button>
    <button id="stopButton" onclick="stopRecording()" disabled>Stop Recording</button>
    <p id="status">Press "Start Recording" to begin.</p>
    <textarea id="audioData" style="display:none;"></textarea>
    <form action="#" method="POST" id="uploadForm">
        <button id="uploadAudio" type="submit" style="display:none;">Upload</button>
    </form>
    """,
    unsafe_allow_html=True
)

# Handle audio data submission
if st.form_submit_button('Upload Audio'):
    audio_data = st.session_state.get('audio_data', None)
    if audio_data:
        audio_bytes = process_audio(audio_data)
        st.audio(audio_bytes, format='audio/wav')
        st.success("Audio file uploaded and processed successfully.")
    else:
        st.warning("No audio data found. Please record audio first.")
else:
    st.warning("Please record audio first.")
