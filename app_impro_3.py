import streamlit as st
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.embeddings import Embeddings

# Assuming you have already set up these
# vector_db = Your vector database object
# embeddings = Your embeddings object

def add_feedback_to_rag(feedback_text, original_query, vector_db, embeddings):
    # Create a new dataframe with the feedback
    feedback_df = pd.DataFrame({
        'query': [original_query],
        'combined_text': [feedback_text]
    })

    # Convert the dataframe to LangChain documents
    feedback_loader = DataFrameLoader(feedback_df, page_content_column='combined_text')
    feedback_documents = feedback_loader.load()

    # Generate embeddings for the feedback documents
    feedback_embeddings = embeddings.embed_documents([doc.page_content for doc in feedback_documents])

    # Add the new documents and their embeddings to the FAISS vector store
    vector_db.add_documents(feedback_documents)

# Initialize chat history and feedback state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'awaiting_feedback' not in st.session_state:
    st.session_state.awaiting_feedback = False

if 'last_prompt' not in st.session_state:
    st.session_state.last_prompt = ""

# Function to display the feedback form
def display_feedback_form():
    feedback_text = st.text_area("Please provide the improved answer:")
    if st.button("Submit Feedback"):
        if feedback_text:
            add_feedback_to_rag(feedback_text, st.session_state.last_prompt, vector_db, embeddings)
            st.success("Thank you for your feedback!")
            st.session_state.awaiting_feedback = False
        else:
            st.error("Please provide the improved answer before submitting.")

# Chat Input
if not st.session_state.awaiting_feedback:
    if prompt := st.chat_input("Was für einen Espresso suchst du?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        response = chain({"question": prompt})
        answer = response['answer']
        
        # Detect and replace URL in the answer
        answer = detect_and_replace_url(answer)

        # Add response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.markdown(answer, unsafe_allow_html=True)
        
        # Save the updated conversation to the database
        save_conversations_to_db(st.session_state.messages, session_id)
        
        # Store the prompt and set awaiting feedback state
        st.session_state.last_prompt = prompt
        st.session_state.awaiting_feedback = True

        # Display feedback options
        st.radio("Do you want to improve this answer?", ('No', 'Yes'))

else:
    display_feedback_form()

# (Optional) Debugging: Print the detected URL and slug
if 'detected_url' in st.session_state:
    st.write(f"Detected URL: {st.session_state.detected_url}")
if 'detected_slug' in st.session_state:
    st.write(f"Detected Slug: {st.session_state.detected_slug}")
