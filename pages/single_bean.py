import streamlit as st
import mysql.connector
import pandas as pd
import urllib.parse
import plotly.express as px
from urllib.parse import urlencode

# Custom CSS
css = """
<style>
section[data-testid="stSidebar"][aria-expanded="true"]{
            display: none;
</style>
"""
# Inject CSS into the Streamlit app
st.markdown(css, unsafe_allow_html=True)

# Fetch query parameters from the URL
query_params = st.experimental_get_query_params()

# Get the language parameter
set_language = query_params.get('set_language', ['de'])[0]   # 'de' as default if not existing in URL
st.session_state.set_language = set_language

# Use the set_language variable as needed in your application for debugging
#st.write(f"The current language is: {set_language}")


### Translations

if st.session_state.set_language == "de":
    single_bean_headline = "Deine persönliche Bohnenempfehlung"
    back_to_chat = "Zurück zum Chat"
    radar_headline = "Diese Bohne im Vergleich zum Mittelwert aller Bohnen"
    detail_description = "Ausführliche Beschreibung"
    degree_of_roasting = "Röstgrad"
    amount_of_crema = "Cremabildung"
    appearance_of_beans = "Bohnenbild"
    caffeine_level = "Koffeingehalt"
    suitability_for_coffee_machines = "Eignung für Vollautomaten"
    mean_values = "Mittelwerte"

else:
    single_bean_headline = "Your personal coffee bean recommendation"
    back_to_chat = "Back to chat"
    radar_headline = "This bean compared to the mean values of all beans"
    detail_description = "Original description (German)"
    degree_of_roasting = "Degree of Roasting"
    amount_of_crema = "Amount of Crema"
    appearance_of_beans = "Appearance of Beans"
    caffeine_level = "Caffeine Level"
    suitability_for_coffee_machines = "Suitability for Automatic Coffee Machines"
    mean_values = "Mean Values"


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

# Function to fetch and calculate means of all beans
def fetch_and_calculate_means():
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    
    # SQL query to select all rows
    cursor.execute("""SELECT 
    roestgrad_num, cremabildung_num, bohnenbild_num, koffeingehalt_num, vollautomaten_num
    FROM beans_info
    """)
    
    # Fetch all rows
    data = cursor.fetchall()
    conn.close()
    
    if data:
        # Convert fetched data to DataFrame
        df = pd.DataFrame(data)
        # Define the columns to calculate means for
        columns_to_calculate = ['roestgrad_num', 'cremabildung_num', 'bohnenbild_num', 'koffeingehalt_num', 'vollautomaten_num']
        # Calculate means for specified columns
        means = df[columns_to_calculate].mean()
        # Ensure means are float
        means = means.astype(float)
        # Create DataFrame with the means
        means_df = pd.DataFrame([means], columns=columns_to_calculate)
        return means_df
    else:
        return pd.DataFrame()  # Return an empty DataFrame if no data is found

# Function to display a single beans_info on the subpage
def display_single_beans_info(source_url):
    # Create a link to the main page with the session_id and set_language
    if session_id:
        params = {
            'session_id': st.session_state.session_id,
            'set_language': st.session_state.set_language
        }
        main_page_url = f"/?{urlencode(params)}"
    else:
        main_page_url = "/"
  
    # Add content and a link to navigate back to the main page
    st.title(single_bean_headline)
    st.markdown(f'<a href="{main_page_url}" target="_self">{back_to_chat}</a>', unsafe_allow_html=True)

    # Fetch single beans_info from the database
    beans_info = fetch_single_beans_info_from_db(source_url)

    # Display beans_info if it exists
    if beans_info:
        st.markdown(f"**Espresso:** {beans_info['title']}")
        st.markdown(f"**Shopping URL:** {beans_info['source_url']}")


        ######## start radar #######
        
        roestgrad_chart = beans_info['roestgrad_num']
        cremabildung_chart = beans_info['cremabildung_num']
        bohnenbild_chart = beans_info['bohnenbild_num']
        koffeingehalt_chart = beans_info['koffeingehalt_num']
        vollautomaten_chart = beans_info['vollautomaten_num']
          
        # Create the DataFrame
        radar_df = pd.DataFrame({
            'r': [roestgrad_chart, cremabildung_chart, bohnenbild_chart, koffeingehalt_chart, vollautomaten_chart],
            'theta': [degree_of_roasting, amount_of_crema, appearance_of_beans, caffeine_level, suitability_for_coffee machines]
        })
        # Convert 'r' column to numeric
        radar_df['r'] = pd.to_numeric(radar_df['r'], errors='coerce')
        
        # Fetch and prepare means data
        means_df = fetch_and_calculate_means()

        if not means_df.empty:
           means_values = means_df.iloc[0].tolist()
           means_radar_df = pd.DataFrame({
               'r': means_values,
               'theta': [degree_of_roasting, amount_of_crema, appearance_of_beans, caffeine_level, suitability_for_coffee_machines]
           })
           means_radar_df['r'] = pd.to_numeric(means_radar_df['r'], errors='coerce')
        else:
           means_radar_df = pd.DataFrame()  # Empty DataFrame if no means data

        # Create radar chart
        fig = px.line_polar(radar_df, r='r', theta='theta', line_close=True, title=radar_headline)

        if not means_radar_df.empty:
           fig.add_scatterpolar(
               r=means_radar_df['r'],
               theta=means_radar_df['theta'],
               fill='toself',
               name= mean_values,
               line=dict(color='#b0896c')
           )

        # Update layout to set the range of the radial axis
        fig.update_layout(
           polar=dict(
               radialaxis=dict(
                   visible=True,
                   range=[0, 6]  # Adjust this range to cover the maximum value in your data
               ),
           ),
           showlegend=True
        )

        # Show the chart in Streamlit
        st.plotly_chart(fig)
       
                        
        ######## end rader ########
                
        if beans_info['review_count'] > 0:
            chart_data_rating = pd.DataFrame({"Rating": [beans_info['rating_value']]})
            st.bar_chart(chart_data_rating, y="Rating", horizontal=True)
            st.markdown(f"**Reviews:** {beans_info['review_count']}")
        else:
            st.markdown("**Reviews:** No reviews yet.")    
                
        st.markdown("---")
        st.markdown(f"**{detail_description}** {beans_info['description']}")
    else:
        st.markdown("**Error:** Bohne not found.")

# Main function to run the Streamlit app
if __name__ == "__main__":
    query_params = st.experimental_get_query_params()
    source_url = query_params.get('url', [None])[0]
    session_id = query_params.get('session_id', [None])[0]
    st.session_state.session_id = session_id
    set_language = query_params.get('set_language', [None])[0]
    st.session_state.set_language = set_language

    if source_url:
        # Decode the URL from the query parameters
        decoded_slug = urllib.parse.unquote(source_url)
        if decoded_slug == None:
            decoded_slug = "nannini-classica-bohne"
        external_url = "https://www.kaffeezentrale.de/"
        decoded_url = external_url + decoded_slug
        
        # Display the beans information
        display_single_beans_info(decoded_url)
    else:
        st.markdown("**Error:** No URL detected in the query parameters.")
    
    # Create a link to the main page with the session_id and set_language
    if session_id:
        params = {
            'session_id': st.session_state.session_id,
            'set_language': st.session_state.set_language
        }
        main_page_url = f"/?{urlencode(params)}"
    else:
        main_page_url = "/"
  
    # Add a link to navigate back to the main page
    st.markdown(f'<a href="{main_page_url}" target="_self">{back_to_chat}</a>', unsafe_allow_html=True)

