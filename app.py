import streamlit as st
import base64
import io
import soundfile as sf

st.title("Audio Recorder and Transcriber")

# Placeholder for recording status
recording_status = st.empty()

# JavaScript code for audio recording
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
if st.button("Upload Audio", key="upload"):
    audio_data = st.session_state.get('audio_data', None)
    if audio_data:
        audio_data = audio_data.split(",")[1]
        audio_bytes = base64.b64decode(audio_data)
        
        audio_file = io.BytesIO(audio_bytes)
        data, samplerate = sf.read(audio_file)
        
        # Display audio player
        st.audio(audio_bytes, format='audio/wav')
        
        st.success("Audio file uploaded and processed successfully.")
    else:
        st.warning("No audio data found. Please record audio first.")

# Hidden form to capture audio data
st.form(key='upload_form', clear_on_submit=True).form_submit_button(
    label='hidden_upload', on_click=lambda: st.session_state.update(audio_data=st.experimental_get_query_params().get('audioData', [None])[0])
)
