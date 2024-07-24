import os
import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from pydub import AudioSegment
import tempfile
import shutil

import matplotlib.animation as animation
import threading

# Audio
from streamlit_mic_recorder import mic_recorder, speech_to_text

# Chatbot imports
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Connection to HuggingFace
huggingface_token = st.secrets["df_token"]
login(token=huggingface_token)

# This info is at the top of each HuggingFace model page
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# Initialize HuggingFace embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "coffee_content/embeddings"
load_path = "coffee_content/faiss_index"
os.makedirs(embeddings_folder, exist_ok=True)
os.makedirs(load_path, exist_ok=True)

embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Check if FAISS files exist before loading
faiss_index_file = os.path.join(load_path, "index.faiss")
faiss_pkl_file = os.path.join(load_path, "index.pkl")

if os.path.exists(faiss_index_file) and os.path.exists(faiss_pkl_file):
    vector_db = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
else:
    st.error(f"FAISS index files not found at {load_path}. Ensure both index.faiss and index.pkl are present.")

# Initialize memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=_llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# Prompt template
input_template = """Answer the question based only on the following context.
Keep your answers short and succinct, but always use whole sentences.
All answers in German.
Important: Always add the link

Previous conversation:
{chat_history}

Context to answer question:
{context}

Question to be answered: {question}
Response:"""

prompt = PromptTemplate(template=input_template, input_variables=["context", "question"])

chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=False, combine_docs_chain_kwargs={"prompt": prompt})

# Streamlit app title
st.title("Welcome to the Beanius, your Espresso expert.")
st.markdown("Just give me a minute, I will be right with you.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

################

state = st.session_state

if 'text_received' not in state:
    state.text_received = []

if 'recording' not in state:
    state.recording = False

if 'stream_data' not in state:
    state.stream_data = np.zeros(44100 * 5)

# Function to update the plot
def update_plot(frame, line, stream_data, samplerate):
    line.set_ydata(stream_data[-samplerate * 5:])
    return line,

# Function to handle audio callback
def audio_callback(indata, frames, time, status):
    if state.recording:
        state.stream_data = np.append(state.stream_data, indata[:, 0])[-samplerate * 5:]

c1, c2 = st.columns(2)
with c1:
    st.write("Was für einen Espresso suchst du?")
with c2:
    if st.button("Start Recording"):
        state.recording = True
        samplerate = 44100  # Hertz

        fig, ax = plt.subplots()
        x = np.linspace(0, 5, samplerate * 5)
        line, = ax.plot(x, np.zeros(samplerate * 5))
        ax.set_ylim([-1, 1])
        ax.set_xlim([0, 5])
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

        stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=samplerate)
        stream.start()

        def run_animation():
            ani = animation.FuncAnimation(fig, update_plot, fargs=(line, state.stream_data, samplerate), interval=50)
            st.pyplot(fig)

        # Run the animation in a separate thread to avoid blocking
        animation_thread = threading.Thread(target=run_animation)
        animation_thread.start()

        # Call speech_to_text to handle the recording and transcription
        text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')
        st.write(text_from_speech)

        state.recording = False
        stream.stop()
        stream.close()

    if st.button("Stop Recording"):
        state.recording = False

transcription = text_from_speech

# Use the transcription as input to the chatbot
if transcription:
    # Display user message in chat message container
    st.chat_message("user").markdown(transcription)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": transcription})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Grinding an answer..."):
        # Send question to chain to get answer
        answer = chain(transcription)

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

################
# Chat Input
if prompt := st.chat_input("Was für einen Espresso suchst du?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chain({"question": prompt})
    msg = response['answer']

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": msg})
    with st.chat_message("assistant"):
        st.markdown(msg)
