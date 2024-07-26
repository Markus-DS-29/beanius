import streamlit as st
import mysql.connector
import pandas as pd

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
    
    cursor.execute('SELECT title, source_url, rating_value, review_count FROM beans_info')
    beans_infos = cursor.fetchall()
    
    cursor.close()
    conn.close()
    
    return beans_infos

# Function to display beans_infos on the subpage
def display_beans_infos():
    st.title("Unsere Bohnen")

    # Fetch conversations from the database
    beans_infos = fetch_beans_infos_from_db()

    # Display beans_infos
    for beans_info in beans_infos:
        st.markdown(f"**Bohne:** {beans_info['title']}")
        st.markdown(f"**URL:** {beans_info['source_url']}")
        st.markdown(f"**Rating:** {beans_info['rating_value']}")
        st.markdown(f"**Reviews:** {beans_info['review_count']}")

        chart_data = pd.DataFrame(
            {
                "Rating": {beans_info['rating_value']},
                "Reviews": {beans_info['review_count']},
            }
        )
        st.bar_chart(chart_data)
        st.markdown("---")

# Main function to run the Streamlit app
if __name__ == "__main__":
    display_beans_infos()
