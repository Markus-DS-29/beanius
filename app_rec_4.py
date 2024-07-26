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
import uuid
from urllib.parse import urlencode


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


# Database connection configuration
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
def save_conversations_to_db(messages, session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    for message in messages:
        cursor.execute('''
            INSERT INTO conversations (timestamp, role, content, session_id)
            VALUES (%s, %s, %s, %s)
        ''', (datetime.now(), message['role'], message['content'], session_id))
    
    conn.commit()
    cursor.close()
    conn.close()

# Function to fetch conversations from the database
def fetch_conversations_from_db(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('''
        SELECT timestamp, role, content
        FROM conversations
        WHERE session_id = %s
        ORDER BY timestamp
    ''', (session_id,))
    conversations = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return conversations

# Function to detect and replace URLs in the answer
def detect_and_replace_url(answer):
    url_pattern = re.compile(r'(https?://\S+)')
    base_url = "https://www.kaffeezentrale.de/"
    urls = url_pattern.findall(answer)
    if urls:
        detected_url = urls[0].rstrip('>,)')
        if detected_url.startswith(base_url):
            detected_slug = detected_url[len(base_url):]
        else:
            detected_slug = None
        # Existing URL parameters
        existing_params = {'url': detected_slug}
        # Add session_id to existing parameters
        existing_params['session_id'] = st.session_state.session_id
        # Generate URL with both parameters
        subpage_url = f"/single_bean?{urlencode(existing_params)}"
        answer = url_pattern.sub(f'<a href="{subpage_url}" target="_self">Hier klicken für mehr Infos.</a>', answer)
        #answer = url_pattern.sub(f'[Info]({subpage_url})', answer)
        
    else:
        detected_url = None
        detected_slug = None
    
    # Store in session state
    st.session_state.detected_url = detected_url
    st.session_state.detected_slug = detected_slug
    
    return answer

# Connection to HuggingFace
huggingface_token = st.secrets["api_keys"]["df_token"]
login(token=huggingface_token)

# HuggingFace model and embeddings
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

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
Most Important: Always add the according source_url to your answer. 

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

# Extract session_id from the URL
query_params = st.experimental_get_query_params()
session_id = query_params.get('session_id', [None])[0]

if session_id:
    # Fetch and load the conversation from the database
    st.session_state.messages = fetch_conversations_from_db(session_id)
else:
    # Generate a new session_id if none exists
    session_id = str(uuid.uuid4())
    st.session_state.session_id = session_id

# Initialize chat history if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

state = st.session_state

if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Was für einen Espresso suchst du?")
with c2:
    text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

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

        # Detect and replace URL in the answer
        answer = detect_and_replace_url(answer)

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save the updated conversation to the database
    save_conversations_to_db(st.session_state.messages, session_id)

# Chat Input
if prompt := st.chat_input("Was für einen Espresso suchst du?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    response = chain({"question": prompt})
    answer = response['answer']
    
    # Detect and replace URL in the answer
    answer = detect_and_replace_url(answer)

    # Add response to chat history
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer, unsafe_allow_html=True)
       
    # Save the updated conversation to the database
    save_conversations_to_db(st.session_state.messages, session_id)
    
    # (Optional) Debugging: Print the detected URL and slug
    if 'detected_url' in st.session_state:
        st.write(f"Detected URL: {st.session_state.detected_url}")
    if 'detected_slug' in st.session_state:
        st.write(f"Detected Slug: {st.session_state.detected_slug}")
