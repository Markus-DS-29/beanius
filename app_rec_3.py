import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from pydub import AudioSegment
import tempfile
import shutil
import time

# Audio
from streamlit_mic_recorder import speech_to_text

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

# Model setup
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

def simulate_recording():
    # Simulate recording process with a delay
    time.sleep(5)
    # Normally here you would call speech_to_text and return its result
    return "Simulated speech text"

c1, c2 = st.columns(2)
with c1:
    st.write("Was für einen Espresso suchst du?")
with c2:
    if st.button("Start Recording"):
        with st.spinner("Recording in progress..."):
            # Simulate the recording process
            state.text_received = simulate_recording()
            # In a real application, replace simulate_recording() with:
            # state.text_received = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

text_from_speech = state.get("text_received", "")

################        

transcription = text_from_speech

# Use the transcription as input to the chatbot
if transcription:
    st.chat_message("user").markdown(transcription)

    st.session_state.messages.append({"role": "user", "content": transcription})

    with st.spinner("Grinding an answer..."):
        answer = chain(transcription)
        response = answer["answer"]

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

################
# Chat Input
if prompt := st.chat_input("Was für einen Espresso suchst du?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    response = chain({"question": prompt})
    msg = response['answer']

    st.session_state.messages.append({"role": "assistant", "content": msg})
    with st.chat_message("assistant"):
        st.markdown(msg)
