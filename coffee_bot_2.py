#streamlit run coffee_bot_2.py


import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import matplotlib.pyplot as plt
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import soundfile as sf
from pydub import AudioSegment

# Chatbot imports
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

# Connection to HuggingFace
huggingface_token = "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"
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

# Speech Recorder and Transcriber
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'recorded_data' not in st.session_state:
    st.session_state.recorded_data = None
if 'fs' not in st.session_state:
    st.session_state.fs = 16000  # Sample rate required for Wav2Vec2
if 'transcription' not in st.session_state:
    st.session_state.transcription = None

# Display recording status
status = st.empty()

# Function to start recording
def start_recording():
    st.session_state.recording = True
    st.session_state.recorded_data = None
    status.text('Recording...')
    duration = 5  # seconds
    st.session_state.recorded_data = sd.rec(int(duration * st.session_state.fs), samplerate=st.session_state.fs, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished
    st.session_state.recording = False
    status.text('Recording finished')

# Function to stop recording
def stop_recording():
    st.session_state.recording = False

# Layout for recording buttons
col1, col2 = st.columns([1, 1])
with col1:
    if st.button('Start Recording'):
        start_recording()
with col2:
    if st.button('Stop Recording') and st.session_state.recording:
        stop_recording()

# Save and plot recorded data
if st.session_state.recorded_data is not None:
    wav_filename = 'recorded_speech.wav'
    
    # Save the WAV file with 16-bit PCM
    with wave.open(wav_filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16 bits = 2 bytes
        wf.setframerate(st.session_state.fs)
        wf.writeframes(st.session_state.recorded_data.tobytes())
    
    st.audio(wav_filename, format='audio/wav')
    st.success('Recording saved successfully!')

    # Plot the recorded data
    fig, ax = plt.subplots()
    ax.plot(st.session_state.recorded_data)
    ax.set_title("Recorded Audio")
    st.pyplot(fig)

    # Transcribe the audio
    if st.button('Transcribe Audio'):
        def preprocess_audio(file_path, output_path, sample_rate=16000):
            audio = AudioSegment.from_file(file_path)
            audio = audio.set_frame_rate(sample_rate)  # Resample to 16 kHz
            audio = audio.set_channels(1)  # Ensure mono channel
            audio.export(output_path, format="wav")

        # Load the model and processor
        audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")
        audio_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-xlsr-53-german")

        # Path to the audio file
        file_path = wav_filename
        processed_file_path = "processed_audio.wav"
        preprocess_audio(file_path, processed_file_path)

        # Load the processed audio file
        speech, sample_rate = sf.read(processed_file_path)

        # Display some information about the audio file
        st.write(f"Sample Rate: {sample_rate} Hz")
        st.write(f"Number of Samples: {len(speech)}")

        # Normalize the audio data
        speech = (speech - np.mean(speech)) / np.std(speech)

        # Ensure the audio input has a minimum length
        min_length = 32000  # 2 seconds of audio at 16000 Hz
        if len(speech) < min_length:
            # Pad the audio to the minimum length
            pad_length = min_length - len(speech)
            speech = np.pad(speech, (0, pad_length), mode='constant')

        # Process the audio data
        input_values = audio_processor(speech, return_tensors="pt", sampling_rate=16000).input_values

        # Perform speech-to-text with the model
        with torch.no_grad():
            logits = audio_model(input_values).logits

        # Decode the predictions to get the transcription
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = audio_processor.batch_decode(predicted_ids)[0]

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


