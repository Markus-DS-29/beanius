import streamlit as st
import base64
import io
import soundfile as sf

st.title("Audio Recorder and Transcriber")

# Function to process audio data
def process_audio(audio_data):
    audio_data = audio_data.split(",")[1]
    audio_bytes = base64.b64decode(audio_data)
    
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    
    # Process the audio data (e.g., transcription)
    # Here you can save the audio file or send it to a transcription service
    
    return audio_bytes

# UI components
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
                    document.getElementById('uploadAudio').click();
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
    <input type="hidden" id="audioData">
    <button id="uploadAudio" style="display:none;" onclick="handleUpload()">Upload</button>
    """,
    unsafe_allow_html=True
)

# Process the audio data
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

if st.session_state.audio_data:
    audio_bytes = process_audio(st.session_state.audio_data)
    
    # Display audio player
    st.audio(audio_bytes, format='audio/wav')
    
    st.success("Audio file uploaded and processed successfully.")
else:
    st.warning("Please record audio first.")

# Form to capture audio data
with st.form(key='upload_form'):
    st.session_state.audio_data = st.text_input('audio_data', value="", type="hidden")
    st.form_submit_button(label='Upload Audio', on_click=lambda: st.session_state.update(audio_data=st.session_state.audio_data))
