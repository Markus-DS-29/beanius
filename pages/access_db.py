import streamlit as st
import mysql.connector


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

# Function to fetch beans_infos from the database
def fetch_beans_infos_from_db():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('SELECT title, source_url FROM beans_info')
    beans_infos = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return beans_infos

# Function to display beans_infos on the subpage
def display_beans_infos():
    st.title("Unsere Bohnen")

    # Fetch conversations from the database
    beans_infos = fetch_beans_infos_from_db()

    # Display conversations
    for conversation in conversations:
        st.markdown(f"**Bohne:** {beans_info['title']}")
        st.markdown(f"**URL:** {conversation['source_url']}")
        st.markdown("---")

# Main function to run the Streamlit app
if __name__ == "__main__":
    display_beans_infos()
