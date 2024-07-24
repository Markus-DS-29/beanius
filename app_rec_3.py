import os
import streamlit as st
import mysql.connector
from datetime import datetime

# Database connection function
def get_db_connection():
    db_config = {
        'host': st.secrets["mysql"]["host"],
        'user': st.secrets["mysql"]["user"],
        'password': st.secrets["mysql"]["password"],
        'database': st.secrets["mysql"]["database"]
    }
    return mysql.connector.connect(**db_config)

# Function to save conversations to the database
def save_conversations_to_db(conversations, session_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Check if session_id already exists in the database
    cursor.execute("SELECT COUNT(*) FROM conversations WHERE session_id = %s", (session_id,))
    if cursor.fetchone()[0] > 0:
        # Session already saved, no need to save again
        cursor.close()
        conn.close()
        return

    # Insert new session data
    for message in conversations:
        query = "INSERT INTO conversations (session_id, role, content, timestamp) VALUES (%s, %s, %s, %s)"
        values = (session_id, message["role"], message["content"], datetime.now())
        cursor.execute(query, values)
    
    conn.commit()
    cursor.close()
    conn.close()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(datetime.now().timestamp())  # Unique session ID based on timestamp

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

# Recording input
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
    st.chat_message("user").markdown(transcription)
    st.session_state.messages.append({"role": "user", "content": transcription})

    with st.spinner("Grinding an answer..."):
        answer = chain(transcription)
        response = answer["answer"]
        st.chat_message("assistant").markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Save conversations once per session
save_conversations_to_db(st.session_state.messages, st.session_state.session_id)

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
