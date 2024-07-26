import streamlit as st
import mysql.connector
import pandas as pd
import urllib.parse

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

# Function to fetch a single beans_info from the database based on its source_url
def fetch_single_beans_info_from_db(source_url):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('SELECT title, source_url, rating_value, review_count, description FROM beans_info WHERE source_url = %s', (source_url,))
    beans_info = cursor.fetchone()
    
    #cursor.close()
    conn.close()
    
    return beans_info

# Function to display a single beans_info on the subpage
def display_single_beans_info(source_url):
    st.title("Unsere Bohne")

    # Fetch single beans_info from the database
    beans_info = fetch_single_beans_info_from_db(source_url)

    # Display beans_info if it exists
    if beans_info:
        st.markdown(f"**Bohne:** {beans_info['title']}")
        st.markdown(f"**URL:** {beans_info['source_url']}")

        if beans_info['review_count'] > 0:
            chart_data_rating = pd.DataFrame({"Rating": [beans_info['rating_value']]})
            st.bar_chart(chart_data_rating, y="Rating", horizontal=True)
            st.markdown(f"**Reviews:** {beans_info['review_count']}")
        else:
            st.markdown("**Reviews:** No reviews yet.")
        
        st.markdown("---")
        st.markdown(f"**Beschreibung:** {beans_info['description']}")
    else:
        st.markdown("**Error:** Bohne not found.")

# Main function to run the Streamlit app
if __name__ == "__main__":
    query_params = st.experimental_get_query_params()
    source_url = query_params.get('url', [None])[0]
    if source_url:
        # Decode the URL from the query parameters
        decoded_url = urllib.parse.unquote(source_url)
        display_single_beans_info(decoded_url)
    else:
        st.markdown("**Error:** No URL detected in the query parameters.")
