import os
import streamlit.components.v1 as components

def st_audiorec():
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    return components.declare_component("st_audiorec", path=build_dir)
import sys
import streamlit as st
from st_audiorec import st_audiorec  # Import the st_audiorec function

print(sys.path)  # Check if the path includes your repository
print(dir(st_audiorec))  # List available attributes in st_audiorec

# Initialize the st_audiorec component
audio_recorder = st_audiorec()

# Use the component in your Streamlit app
st.title('Streamlit App with Audio Recorder')

# Call the component function in your Streamlit app
wav_audio_data = audio_recorder()  # Call the component to render it

# Display or process the audio data as needed
if wav_audio_data:
    st.audio(BytesIO(wav_audio_data), format='audio/wav')
