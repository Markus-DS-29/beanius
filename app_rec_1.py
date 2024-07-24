import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text

state = st.session_state

if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Convert speech to text:")
with c2:
    text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

for text in state.text_received:
    st.text(text_from_speech)
