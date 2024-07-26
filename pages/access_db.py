simport streamlit as st
import mysql.connector
from datetime import datetime

# Function to establish a connection to the database using Streamlit secrets
def get_db_connection():
    db_config = {
        'user': st.secrets["mysql"]["DB_USER"],
        'password': st.secrets["mysql"]["DB_PASS"],
        'host': st.secrets["mysql"]["host"],
        'database': st.secrets["mysql"]["BEANS_DATABASE"]
    }
    
    conn = mysql.connector.connect(**db_config)
    return conn

# Function to fetch conversations from the database
def fetch_conversations_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('SELECT title , source_url FROM beans_info')
    conversations = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return conversations

# Function to display conversations on the subpage
def display_conversations():
    st.title("Unsere Bohnen")

    # Fetch conversations from the database
    conversations = fetch_conversations_from_db()

    # Display conversations
    for conversation in conversations:
        st.markdown(f"**Bohne:** {beans_info['title']}")
        st.markdown(f"**Link:** {beans_info['source_url']}")
        st.markdown("---")

# Main function to run the Streamlit app
if __name__ == "__main__":
    display_conversations()
