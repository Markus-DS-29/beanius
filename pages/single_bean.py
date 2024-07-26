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

# Function to fetch a single beans_info from the database based on its index
def fetch_single_beans_info_from_db(index_id):
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    cursor.execute('SELECT title, source_url, rating_value, review_count FROM beans_info WHERE id = %s', (index_id,))
    beans_info = cursor.fetchone()
    
    cursor.close()
    conn.close()
    
    return beans_info

# Function to display a single beans_info on the subpage
def display_single_beans_info(index_id):
    st.title("Unsere Bohne")

    # Fetch single beans_info from the database
    beans_info = fetch_single_beans_info_from_db(index_id)

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
    else:
        st.markdown("**Error:** Bohne not found.")

# Main function to run the Streamlit app
if __name__ == "__main__":
    single_product_index_id = st.number_input("Enter the product index ID", min_value=1)
    if single_product_index_id:
        display_single_beans_info(single_product_index_id)
