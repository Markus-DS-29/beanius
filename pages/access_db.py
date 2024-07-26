import streamlit as st
from datetime import datetime
import mysql.connector

# Function to establish a connection to the database
def get_db_connection():
    conn = mysql.connector.connect(
        host="your_host",
        user="your_user",
        password="your_password",
        database="your_database"
    )
    return conn

# Function to fetch conversations from the database
def fetch_conversations_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('SELECT * FROM conversations ORDER BY timestamp DESC')
    conversations = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return conversations




# Import the function to fetch conversations
from your_module import fetch_conversations_from_db

def display_conversations():
    st.title("Conversation History")

    # Fetch conversations from the database
    conversations = fetch_conversations_from_db()

    # Display conversations
    for conversation in conversations:
        st.markdown(f"**Timestamp:** {conversation['timestamp']}")
        st.markdown(f"**Role:** {conversation['role']}")
        st.markdown(f"**Content:** {conversation['content']}")
        st.markdown("---")

if __name__ == "__main__":
    display_conversations()
