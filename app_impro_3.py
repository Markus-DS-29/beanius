import os
import pandas as pd
import streamlit as st
import mysql.connector
from datetime import datetime
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

# Custom CSS
css = """
<style>
section[data-testid="stSidebar"] {
    display: none;
}
</style>
"""
# Inject CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)

# Initialize chat history and feedback state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False

if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""

if 'improved_answer' not in st.session_state:
    st.session_state.improved_answer = ""

if 'query_data' not in st.session_state:
    st.session_state.query_data = ""

# Database connection configuration for conversations
db_config = {
    'user': st.secrets["mysql"]["user"],
    'password': st.secrets["mysql"]["password"],
    'host': st.secrets["mysql"]["host"],
    'database': st.secrets["mysql"]["database"]
}

# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(**db_config)

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

# Function to fetch content from the database
def fetch_chunks_sql_from_db():
    conn = get_db_connection_2()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM chunks_db')
    chunks_sql = cursor.fetchall()
    cursor.close()
    conn.close()
    return chunks_sql

def fetch_feedback_sql_from_db():
    conn = get_db_connection_2()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT * FROM feedback_db')
    feedback_sql = cursor.fetchall()
    cursor.close()
    conn.close()
    return feedback_sql

# Function to save conversations to the database
def save_conversations_to_db(messages, session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    for message in messages:
        cursor.execute('INSERT INTO conversations (timestamp, role, content, session_id) VALUES (%s, %s, %s, %s)',
                       (datetime.now(), message['role'], message['content'], session_id))
    conn.commit()
    cursor.close()
    conn.close()

# Function to fetch conversations from the database
def fetch_conversations_from_db(session_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute('SELECT timestamp, role, content FROM conversations WHERE session_id = %s ORDER BY timestamp', (session_id,))
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
        existing_params = {'url': detected_slug, 'session_id': st.session_state.session_id}
        subpage_url = f"/single_bean?{urlencode(existing_params)}"
        answer = url_pattern.sub(f'<a href="{subpage_url}" target="_self">Hier klicken für mehr Infos.</a>', answer)
    else:
        detected_url = None
        detected_slug = None
    st.session_state.detected_url = detected_url
    st.session_state.detected_slug = detected_slug
    return answer

# Fetch and combine data from the database
chunks_data = fetch_chunks_sql_from_db()
feedback_data = fetch_feedback_sql_from_db()
chunks_sql_df = pd.DataFrame(chunks_data)
feedback_sql_df = pd.DataFrame(feedback_data)
chunks_text = chunks_sql_df[['combined_text']]
feedback_text = feedback_sql_df[['combined_text']]
all_data_df = pd.concat([chunks_text, feedback_text], ignore_index=True)

# Function to add feedback to the RAG
def add_feedback_to_rag(feedback_text, original_query, vector_db, embeddings):
    feedback_df = pd.DataFrame({'query': [original_query], 'combined_text': [feedback_text]})
    feedback_loader = DataFrameLoader(feedback_df, page_content_column='combined_text')
    feedback_documents = feedback_loader.load()
    feedback_embeddings = embeddings.embed_documents([doc.page_content for doc in feedback_documents])
    vector_db.add_documents(feedback_documents)

# Function to display the feedback form
def display_feedback_form():
    st.session_state.improved_answer = st.text_area("Please provide the improved answer:", key='feedback_text_area')
    if st.button("Submit Feedback", key='submit_feedback'):
        if st.session_state.improved_answer:
            st.session_state.query_data = st.session_state.last_prompt
            st.success("Thank you for your feedback!")
            st.session_state.awaiting_feedback = False
            st.session_state.show_feedback_options = False
        else:
            st.error("Please provide the improved answer before submitting.")

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

# Function to create FAISS vector store
@st.cache(allow_output_mutation=True)
def create_faiss_vector_store(dataframe, embedding_model, embeddings_folder):
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)
    loader = DataFrameLoader(dataframe, page_content_column='combined_text')
    documents = loader.load()
    vector_db = FAISS.from_documents(documents, embeddings)
    return vector_db

vector_db = create_faiss_vector_store(all_data_df, embedding_model, embeddings_folder)
retriever = vector_db.as_retriever(search_kwargs={"k": 1})
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
st.write("FAISS vector store created successfully.")

@st.cache_resource
def init_embeddings():
    return HuggingFaceEmbeddings(model_name=embedding_model)
embeddings = init_embeddings()

@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=_llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# Define the prompt template
input_template = """Answer the question based only on the following context.
Keep your answers short and succinct, but always use whole sentences.
All answers must always be in German!
Most Important: Always add the according url to your answer, no "(" or "'" or ")". 

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

# Extract session_id from the URL if available
query_params = st.experimental_get_query_params()
session_id_from_url = query_params.get('session_id', [None])[0]

if session_id_from_url:
    st.session_state.session_id = session_id_from_url
elif 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

session_id = st.session_state.session_id
st.write(f"Session ID: {session_id}")

# Initialize chat history and fetch conversations
if "messages" not in st.session_state:
    st.session_state.messages = []

if session_id_from_url:
    st.session_state.messages = fetch_conversations_from_db(session_id)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

state = st.session_state
if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Was für einen Espresso suchst du?")
with c2:
    text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

transcription = text_from_speech

if transcription:
    st.chat_message("user").markdown(transcription)
    st.session_state.messages.append({"role": "user", "content": transcription})

    with st.spinner("Grinding an answer..."):
        response = chain({"question": transcription})
        answer = response["answer"]
        answer = detect_and_replace_url(answer)
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer})

    save_conversations_to_db(st.session_state.messages, session_id)

# Handle user input through the chat input field
if not st.session_state.awaiting_feedback:
    prompt = st.chat_input("Was für einen Espresso suchst du?")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = chain({"question": prompt})
        answer = response['answer']
        answer = detect_and_replace_url(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)

        save_conversations_to_db(st.session_state.messages, session_id)
        st.session_state.last_prompt = prompt
        st.session_state.awaiting_feedback = True
        st.session_state.show_feedback_options = True

# Show feedback options and handle feedback submission
if st.session_state.awaiting_feedback:
    if st.session_state.show_feedback_options:
        feedback_option = st.radio("Do you want to improve this answer?", ('No', 'Yes'), key='feedback_radio')
        if feedback_option == 'No':
            st.session_state.awaiting_feedback = False
            st.session_state.show_feedback_options = False
        elif feedback_option == 'Yes':
            st.session_state.show_feedback_options = False
            display_feedback_form()
    else:
        display_feedback_form()

# (Optional) Debugging: Print the query and feedback
if st.session_state.query_data:
    st.write(f"Query: {st.session_state.query_data}")
if st.session_state.improved_answer:
    st.write(f"Improved Answer: {st.session_state.improved_answer}")

# (Optional) Debugging: Print the detected URL and slug
if 'detected_url' in st.session_state:
    st.write(f"Detected URL: {st.session_state.detected_url}")
if 'detected_slug' in st.session_state:
    st.write(f"Detected Slug: {st.session_state.detected_slug}")
