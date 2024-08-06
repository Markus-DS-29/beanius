####### FINAL? Status: Everything seems to be working. This version is for better performance only. ###########

import os
import pandas as pd
import streamlit as st
from streamlit import cache_resource #new
import mysql.connector
from datetime import datetime
import streamlit.components.v1 as components
#import sounddevice as sd
import numpy as np
#import wave
import matplotlib.pyplot as plt
#import torch
#from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
#import soundfile as sf
#from pydub import AudioSegment
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
from langchain.document_loaders import DataFrameLoader
from langchain.chains import RetrievalQA

# Custom CSS
css = """
<style>
section[data-testid="stSidebar"]{
  display: none;
}

.st-emotion-cache-p4micv.eeusbqq0 {
  width: 100px;
  height: 100px;
  margin-right: 30px;
}

.st-emotion-cache-bho8sy.eeusbqq1 {
  width: 100px;
  height: 100px;
  margin-right: 30px;
}

.st-emotion-cache-1ghhuty.eeusbqq1 {
  width: 100px;
  height: 100px;
  margin-right: 30px;
}
</style>
"""

# Inject CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)

#### Language switcher ###

# Initialize session state if it doesn't exist

# Extract session_id from the URL if available
query_params = st.experimental_get_query_params()
set_language_from_url = query_params.get('set_language', [None])[0]
if set_language_from_url:
    # Always use set_language from URL if present
    st.session_state.set_language = set_language_from_url
elif 'set_language' not in st.session_state:
    # Generate a new set_language if none exists
    st.session_state.set_language = "de"

# Display session ID for debugging
st.write(f"Set_Language from URL: {st.session_state.set_language}")

if 'set_language' not in st.session_state:
    st.session_state.set_language = 'de'  # Default value

# Define the callback function to toggle the language
def toggle_language():
    if st.session_state.set_language == 'de':
        st.session_state.set_language = 'en'
    else:
        st.session_state.set_language = 'de'

# Display the current language
st.write(f"Current language: {st.session_state.set_language}")

# Create a button that toggles the language when clicked
if st.button("DE / EN", on_click=toggle_language):
    st.write(f"Language changed to: {st.session_state.set_language}")

### End Language switcher ###    

### translations ###

if st.session_state.set_language == 'de':
    greeting = "Willkommen bei Beanius, deinem Espresso-Experten."
    first_question = "Was für einen Espresso suchst du?"
    start_recording = "Aufnahme starten"
    stop_recording = "Aufnahme beenden"
    grinding = "Die Antwort ist in der Mühle..."
else: 
    greeting = "Welcome to Beanius, your Espresso expert."
    first_question = "What kind of espresso are you looking for?" 
    start_recording = "Start recording"
    stop_recording = "Stop recording"
    grinding = "Grinding an answer..."





# Initialize chat history and feedback state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'query_data' not in st.session_state:
    st.session_state.query_data = ""

### Get and save RAG-data from database ###

# Database connection configuration for RAG data
db_config_data = {
    'user': st.secrets["mysql_data"]["user"],
    'password': st.secrets["mysql_data"]["password"],
    'host': st.secrets["mysql_data"]["host"],
    'database': st.secrets["mysql_data"]["database_2"]
}

# Function to connect to the database for RAG data 
def get_db_connection_2():
    return mysql.connector.connect(**db_config_data)

# Function to fetch RAG-chunks from the database
def fetch_chunks_sql_from_db():
    conn = get_db_connection_2()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT *
        FROM chunks_db
    ''')
    chunks_sql = cursor.fetchall()
    cursor.close()
    conn.close()
    return chunks_sql

# Function to fetch RAG-feedback from the database
def fetch_feedback_sql_from_db():
    conn = get_db_connection_2()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('''
        SELECT *
        FROM feedback_db
    ''')
    feedback_sql = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback_sql

### Get and save conversations from database

# Database connection configuration for conversations
db_config = {
    'user': st.secrets["mysql"]["user"],
    'password': st.secrets["mysql"]["password"],
    'host': st.secrets["mysql"]["host"],
    'database': st.secrets["mysql"]["database"]
}

# Function to connect to the database for conversations
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
        detected_url = urls[0].rstrip('>,).')
        if detected_url.startswith(base_url):
            detected_slug = detected_url[len(base_url):]
        else:
            detected_slug = None
        # Existing URL parameters
        existing_params = {'url': detected_slug}
        # Add session_id to existing parameters
        existing_params['session_id'] = st.session_state.session_id
        # Add language to existing parameters
        existing_params['set_language'] = st.session_state.set_language
        # Generate URL with both parameters
        if detected_slug == None:
            answer = url_pattern.sub(f'<a href="/" target="_self">No Link.</a>', answer)
        else:    
            subpage_url = f"/single_bean?{urlencode(existing_params)}"
            answer = url_pattern.sub(f'<a href="{subpage_url}" target="_self">Details & Infos.</a>', answer)
            #answer = url_pattern.sub(f'[Info]({subpage_url})', answer)
    else:
        detected_url = None
        detected_slug = None
    
    # Store in session state
    st.session_state.detected_url = detected_url
    st.session_state.detected_slug = detected_slug
    
    return answer

### Combine data from db ###

# Fetch data from the database
chunks_data = fetch_chunks_sql_from_db()
feedback_data = fetch_feedback_sql_from_db()

# Convert the list of dictionaries to a DataFrame
chunks_sql_df = pd.DataFrame(chunks_data)
feedback_sql_df = pd.DataFrame(feedback_data)

# Extract the 'combined_text' columns from both DataFrames
chunks_text = chunks_sql_df[['combined_text']]
feedback_text = feedback_sql_df[['combined_text']]

# Concatenate the DataFrames vertically
all_data_df = pd.concat([chunks_text, feedback_text], ignore_index=True)


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

### Start FAISS

# Function to create and cache FAISS vector store
@cache_resource
def create_faiss_vector_store(dataframe, embedding_model, embeddings_folder):
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

    # Convert dataframe to LangChain documents
    loader = DataFrameLoader(dataframe, page_content_column='combined_text')
    documents = loader.load()

    # Create FAISS vector store from loader
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

# Create FAISS vector store using function for cache
vector_db = create_faiss_vector_store(all_data_df, embedding_model, embeddings_folder)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
#st.write("FAISS vector store created successfully.")

### Embeddings, memory, prompt

# Embeddings
#@st.cache_resource   #-> caching? 
def init_embeddings():
    return HuggingFaceEmbeddings(model_name=embedding_model)
embeddings = init_embeddings() 

# Initialize memory
#@st.cache_resource   #-> caching?
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=_llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)


### Prompt Language ####

if st.session_state.set_language == "de":
    # German prompt template
    input_template = """Answer the question based only on the following context.
    Extremely important: always stick closely to this prompt template!!
    Keep your answers short and succinct, but always use whole sentences. Don't write "Link:" or similar.
    All answers must always be in German!
    Most Important: Always add the 1 according url to your answer, if it comes from https://www.kaffeezentrale.de/ ! 
    Otherwise don't add any URL.
    Never use any of the following characters in your answer: ( ' ) < > 
    Never add the context to your answer.

    Previous conversation:
    {chat_history}

    Context to answer question:
    {context}

    Question to be answered: {question}
    Response:"""
else:
    # English prompt template
    input_template = """Answer the question based only on the following context.
    Extremely important: always stick closely to this prompt template!!
    Keep your answers short and succinct, but always use whole sentences. Don't write "Link:" or similar.
    All answers must always be in English!
    Most Important: Always add the 1 according url to your answer, if it comes from https://www.kaffeezentrale.de/ ! 
    Otherwise don't add any URL.
    Never use any of the following characters in your answer: ( ' ) < > 
    Never add the context to your answer.

    Previous conversation:
    {chat_history}

    Context to answer question:
    {context}

    Question to be answered: {question}
    Response:"""

prompt = PromptTemplate(template=input_template, input_variables=["context", "question"])

chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=False, combine_docs_chain_kwargs={"prompt": prompt})

# Streamlit app title
st.title(f"{greeting}")
#st.markdown("Just give me a minute, I will be right with you.")

### Extract and handle session_id

# Extract session_id from the URL if available
query_params = st.experimental_get_query_params()
session_id_from_url = query_params.get('session_id', [None])[0]
set_language_from_url = query_params.get('set_language', [None])[0]

if session_id_from_url:
    # Always use session_id from URL if present
    st.session_state.session_id = session_id_from_url
elif 'session_id' not in st.session_state:
    # Generate a new session_id if none exists
    st.session_state.session_id = str(uuid.uuid4())
# Now we can safely use st.session_state.session_id
session_id = st.session_state.session_id

if set_language_from_url:
    # Always use set_language from URL if present
    st.session_state.set_language = set_language_from_url
elif 'set_language' not in st.session_state:
    # Generate a new set_language if none exists
    st.session_state.set_language = "de"

# Display session ID for debugging
st.write(f"Set_Language: {st.session_state.set_language}")

### Initialize chat history and fetch conversations

#load custom avatar images for user and beanius
path_to_user_avatar = "images/user_icon.png"
with open(path_to_user_avatar, "rb") as file:
    user_image = file.read()
path_to_beanius_avatar = "images/beanius_icon.png"
with open(path_to_beanius_avatar, "rb") as file:
    beanius_image = file.read()

# Initialize chat history if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []

# Fetch conversations from the database if session_id is present
if session_id_from_url:
    st.session_state.messages = fetch_conversations_from_db(session_id)

# Display chat messages with avatars
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message(message["role"], avatar=user_image):
            st.markdown(message["content"], unsafe_allow_html=True)
    else:
        with st.chat_message(message["role"], avatar=beanius_image):
            st.markdown(message["content"], unsafe_allow_html=True)


# Display chat messages from history on app rerun
state = st.session_state
if 'text_received' not in state:
    state.text_received = []

### Speech to text

# Display button and call function
c1, c2 = st.columns(2)
with c1:
    st.write(f"{first_question}")
with c2:
    text_from_speech = speech_to_text(start_prompt=start_recording, stop_prompt=stop_recording, language=st.session_state.set_language, use_container_width=True, just_once=True, key='STT')

transcription = text_from_speech

# Use the transcription as input to the chatbot
if transcription:
    # Display user message in chat message container
    with st.chat_message("user", avatar=user_image):
      st.markdown(transcription)
    #st.chat_message("user").markdown(transcription)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": transcription})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner(f"{grinding}"):
        # Send question to chain to get answer
        response = chain({"question": transcription})
        
        # Extract answer from dictionary returned by chain
        answer = response["answer"]

        # Detect and replace URL in the answer
        answer = detect_and_replace_url(answer)

        # Display chatbot response in chat message container
        with st.chat_message("assistant", avatar=beanius_image):
            st.markdown(answer, unsafe_allow_html=True)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})

    # Save the updated conversation to the database
    save_conversations_to_db(st.session_state.messages, session_id)

### Display user input field in addition to speech-to-text

if prompt := st.chat_input(f"{first_question}"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar=user_image):
            st.markdown(prompt)

        # Generate response
        response = chain({"question": prompt})
        answer = response['answer']
        
        # Detect and replace URL in the answer
        answer = detect_and_replace_url(answer)

        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant", avatar=beanius_image):
            st.markdown(answer, unsafe_allow_html=True)
        
        # Save the updated conversation to the database
        save_conversations_to_db(st.session_state.messages, session_id)
