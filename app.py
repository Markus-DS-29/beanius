import streamlit as st
import soundfile as sf
import io

st.title("Audio Recorder and Transcriber")

# Upload the audio file
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Read the audio file
    audio_bytes = uploaded_file.read()
    
    # Display the audio player
    st.audio(audio_bytes, format='audio/wav')
    
    # Save the audio file
    audio_file = io.BytesIO(audio_bytes)
    data, samplerate = sf.read(audio_file)
    
    # Process the audio data (e.g., transcription)
    # Here you can save the audio file or send it to a transcription service
    
    st.write("Audio file uploaded and processed successfully.")
else:
    st.write("Please upload an audio file to continue.")

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
                const audioUrl = URL.createObjectURL(audioBlob);
                const audio = new Audio(audioUrl);
                audio.play();

                const reader = new FileReader();
                reader.readAsDataURL(audioBlob);
                reader.onloadend = function() {
                    const base64data = reader.result;
                    fetch("https://beanius-rec-1.streamlit.app/upload", {
                        method: "POST",
                        body: JSON.stringify({ data: base64data }),
                        headers: {
                            "Content-Type": "application/json"
                        }
                    }).then(response => {
                        if (response.ok) {
                            console.log("Audio uploaded successfully");
                        } else {
                            console.error("Audio upload failed");
                        }
                    }).catch(error => {
                        console.error("Error:", error);
                    });
                };
            });
        } catch (error) {
            console.error('Error accessing microphone:', error);
        }
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
