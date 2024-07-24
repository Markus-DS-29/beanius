import streamlit as st
import streamlit.components.v1 as components

# Streamlit app for recording audio
def audio_recorder():
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

                    // Send the audio blob to the backend
                    const reader = new FileReader();
                    reader.readAsDataURL(audioBlob);
                    reader.onloadend = function() {
                        const base64data = reader.result;
                        fetch("http://localhost:8000/upload", {
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

audio_recorder()
