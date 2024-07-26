import streamlit as st
from datetime import datetime

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
