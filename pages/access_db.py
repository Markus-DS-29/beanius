import streamlit as st
import mysql.connector
from datetime import datetime

# Function to establish a connection to the database using Streamlit secrets
def get_db_connection():
    db_config = {
        'user': st.secrets["mysql"]["user"],
        'password': st.secrets["mysql"]["password"],
        'host': st.secrets["mysql"]["host"],
        'database': st.secrets["mysql"]["database"]
    }
    
    conn = mysql.connector.connect(**db_config)
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

# Function to display conversations on the subpage
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

# Main function to run the Streamlit app
if __name__ == "__main__":
    display_conversations()
