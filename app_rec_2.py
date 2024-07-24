#streamlit run coffee_bot_2.py

import streamlit as st
import os
import requests
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# Audio
from streamlit_mic_recorder import mic_recorder, speech_to_text

# Chatbot imports
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from huggingface_hub import login

#connection to huggingface
huggingface_token = st.secrets["df_token"]
login(token=huggingface_token)

# This info is at the top of each HuggingFace model page
hf_model = "mistralai/Mistral-7B-Instruct-v0.3"
llm = HuggingFaceEndpoint(repo_id=hf_model)

# Initialize HuggingFace embeddings
embedding_model = "sentence-transformers/all-MiniLM-l6-v2"
embeddings_folder = "/coffee_content/"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model, cache_folder=embeddings_folder)

# Read FAISS vector store from local drive
load_path = "coffee_content/faiss_index"
vector_db = FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)
retriever = vector_db.as_retriever(search_kwargs={"k": 5})

# Initialize memory
@st.cache_resource
def init_memory(_llm):
    return ConversationBufferMemory(
        llm=llm,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)
memory = init_memory(llm)

# Prompt template
input_template = """Answer the question based only on the following context.
Keep your answers short and succinct, but always use whole sentences.
All answers in German.
Important: Allways add the link

Previous conversation:
{chat_history}

Context to answer question:
{context}

Question to be answered: {question}
Response:"""

prompt = PromptTemplate(template=input_template, input_variables=["context", "question"])

chain = ConversationalRetrievalChain.from_llm(llm, retriever=retriever, memory=memory, return_source_documents=False, combine_docs_chain_kwargs={"prompt": prompt})

# Streamlit app title
st.title("Welcome to the Beanius, your Espresso expert.")
st.markdown("Just give me a minute, I will be right with you.")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

       
################

state = st.session_state

if 'text_received' not in state:
    state.text_received = []

c1, c2 = st.columns(2)
with c1:
    st.write("Convert speech to text:")
with c2:
    text_from_speech = speech_to_text(language='de', use_container_width=True, just_once=True, key='STT')

#if text_from_speech:
 #   state.text_received.append(text_from_speech)
    
#for text in state.text_received:
 #   st.text(text_from_speech)
st.text(text_from_speech)
        
################

transcription = text_from_speech

# Display the transcription
st.session_state.transcription = transcription
st.write(f"Transcription: {transcription}")

# Use the transcription as input to the chatbot
if transcription:
    # Display user message in chat message container
    st.chat_message("user").markdown(transcription)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": transcription})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Grinding an answer..."):
        # Send question to chain to get answer
        answer = chain(transcription)

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# React to text input
if prompt := st.chat_input("Welche Art Espresso suchst du?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Grinding an answer..."):
        # Send question to chain to get answer
        answer = chain(prompt)

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
