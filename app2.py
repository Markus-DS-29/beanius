import streamlit as st
from st_audiorec import st_audiorec  # Import the st_audiorec function

# Initialize the st_audiorec component
audio_recorder = st_audiorec()

# Debugging prints
print(audio_recorder)  # Check what `audio_recorder` is
print(callable(audio_recorder))  # Check if it's callable

# Call the component function in your Streamlit app
if callable(audio_recorder):
    wav_audio_data = audio_recorder()  # Use the component to render it
else:
    st.error("Error: audio_recorder is not callable")

# Display or process the audio data as needed
if wav_audio_data:
    st.audio(BytesIO(wav_audio_data), format='audio/wav')
