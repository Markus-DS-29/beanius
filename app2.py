import streamlit as st
from st_audiorec import st_audiorec  # Import the st_audiorec function

# Initialize the st_audiorec component
audio_recorder = st_audiorec()

# Use the component in your Streamlit app
st.title('Streamlit App with Audio Recorder')

# Call the component function in your Streamlit app
wav_audio_data = audio_recorder()  # Call the component to render it

# Display or process the audio data as needed
if wav_audio_data:
    st.audio(BytesIO(wav_audio_data), format='audio/wav')
