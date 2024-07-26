#status: Running as Streamlit WebApp and saving to SQL

import os
import streamlit as st
import mysql.connector
from datetime import datetime
import streamlit.components.v1 as components
#import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from pydub import AudioSegment
import tempfile
import shutil
import re


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

# MySQL database connection details from Streamlit secrets
db_config = {
    'user': st.secrets["mysql"]["user"],
    'password': st.secrets["mysql"]["password"],
    'host': st.secrets["mysql"]["host"],
    'database': st.secrets["mysql"]["database"]
}

# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(**db_config)

# Function to save conversations to the database
def save_conversations_to_db(messages):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for message in messages:
        cursor.execute('''
            INSERT INTO conversations (timestamp, role, content)
            VALUES (%s, %s, %s)
        ''', (datetime.now(), message['role'], message['content']))
    
    conn.commit()
    cursor.close()
    conn.close()

# Function to detect and replace URLs in the answer
def detect_and_replace_url(answer):
    url_pattern = re.compile(r'(https?://\S+)')
    base_url = "https://www.kaffeezentrale.de/"
    urls = url_pattern.findall(answer)
    if urls:
        detected_url = urls[0]
        if detected_url.startswith(base_url):
            detected_slug = detected_url[len(base_url):]
        else:
            detected_slug = None
        answer = url_pattern.sub('[Info](/single_bean)', answer)
    else:
        detected_url = None
        detected_slug = None

    # Store in session state
    st.session_state.detected_url = detected_url
    st.session_state.detected_slug = detected_slug
    return answer


# Connection to huggingface
huggingface_token = st.secrets["api_keys"]["df_token"]
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

c1, c2 = st.columns(2)
with c1:
    st.write("Was für einen Espresso suchst du?")
with c2:
    text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

#if text_from_speech:
 #   state.text_received.append(text_from_speech)
    
#for text in state.text_received:
 #   st.text(text_from_speech)
#st.write(text_from_speech)
        
################        

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
        response = chain({"question": transcription})
        
        # Extract answer from dictionary returned by chain
        answer = response["answer"]

        # Detect and replace URL in the answer using function frome above (c/p to text-chat)
        answer, answer_url = detect_and_replace_url(answer)

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save the updated conversation to the database
    save_conversations_to_db(st.session_state.messages)


################
# Chat Input
if prompt := st.chat_input("Was für einen Espresso suchst du?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chain({"question": prompt})
    answer = response['answer']
    
    # Detect and replace URL in the answer using function frome above (c/p from text-chat)
    answer, answer_url = detect_and_replace_url(answer)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Save the updated conversation to the database
    save_conversations_to_db(st.session_state.messages)
