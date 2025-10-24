import streamlit as st
import os
from dotenv import load_dotenv
import sys

sys.path.append("g:\\todoapp")
from ytbot import ask_ytbot

load_dotenv()


st.title("Langchain Chatbot")
youtube_link = st.text_input("Enter YouTube Link:")

if youtube_link:
    st.session_state.youtube_link = youtube_link
    st.write(f"You entered: {youtube_link}")
    from ytbot import generate_vector
    try:
        st.session_state.vector_store = generate_vector(youtube_link)
    except Exception as e:
        error_message = f"An error occurred: {e}\n\n"
        
        st.error(error_message)
        st.stop()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if prompt := st.chat_input("Ask anything"):
    # Display user message in chat message container
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        # Get assistant response from ytbot
        from ytbot import ask_ytbot
        assistant_response = ask_ytbot(st.session_state.messages, st.session_state.get("vector_store"))

        full_response = ""
        for chunk in assistant_response:
            full_response += chunk
            message_placeholder.markdown(full_response + " ▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
