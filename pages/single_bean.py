import streamlit as st
import mysql.connector
import pandas as pd
import urllib.parse
import plotly.graph_objects as go

# Custom CSS
css = """
<style>
section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
</style>
"""
# Inject CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)


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
    
    cursor.execute("""SELECT 
    title, source_url, rating_value, review_count, description, roestgrad_num, cremabildung_num, bohnenbild_num, koffeingehalt_num, vollautomaten_num
    FROM beans_info WHERE source_url = %s
    """, (source_url,))
    beans_info = cursor.fetchone()
    
    conn.close()
    
    return beans_info

# Function to display a single beans_info on the subpage
def display_single_beans_info(source_url):
    st.title("Unsere Bohnenempfehlung")
    # Create a link to the main page with the session_id
    if session_id:
        main_page_url = f"/?session_id={session_id}"
    else:
        main_page_url = "/"
  
    # Add a link to navigate back to the main page
    st.markdown(f'<a href="{main_page_url}" target="_self">Zurück zum Chat</a>', unsafe_allow_html=True)

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

        ######## start radar #######
        categories = ['roestgrad_num', 'cremabildung_num', 'bohnenbild_num', 'koffeingehalt_num', 'vollautomaten_num']

        fig = go.Figure()
            
        fig.add_trace(go.Scatterpolar(
                  r=[roestgrad_num, cremabildung_num, bohnenbild_num, koffeingehalt_num, vollautomaten_num],
                  theta=categories,
                  fill='toself',
                  name='Product A'
        ))
        fig.add_trace(go.Scatterpolar(
                  r=[4, 3, 2.5, 1, 2],
                  theta=categories,
                  fill='toself',
                  name='Product B'
        ))
            
        fig.update_layout(
              polar=dict(
                radialaxis=dict(
                  visible=True,
                  range=[0, 5]
                )),
             showlegend=False
        )
            
        fig.show()

                
        ######## end rader ########
                
        st.markdown("---")
        st.markdown(f"**Beschreibung:** {beans_info['description']}")
    else:
        st.markdown("**Error:** Bohne not found.")

# Main function to run the Streamlit app
if __name__ == "__main__":
    query_params = st.experimental_get_query_params()
    source_url = query_params.get('url', [None])[0]
    session_id = query_params.get('session_id', [None])[0]
    
    if source_url:
        # Decode the URL from the query parameters
        decoded_slug = urllib.parse.unquote(source_url)
        external_url = "https://www.kaffeezentrale.de/"
        decoded_url = external_url + decoded_slug
        
        # Display the beans information
        display_single_beans_info(decoded_url)
    else:
        st.markdown("**Error:** No URL detected in the query parameters.")
    
    # Create a link to the main page with the session_id
    if session_id:
        main_page_url = f"/?session_id={session_id}"
    else:
        main_page_url = "/"
  
    # Add a link to navigate back to the main page
    st.markdown(f'<a href="{main_page_url}" target="_self">Zurück zum Chat</a>', unsafe_allow_html=True)

