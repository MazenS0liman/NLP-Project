import os
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

import sys
import asyncio

# Create a single global loop
_loop = asyncio.new_event_loop()
asyncio.set_event_loop(_loop)

import warnings
import torch
import streamlit as st
torch.classes.__path__ = []

# suppress warnings
warnings.filterwarnings('ignore')

# import custom chatbot modules
from chatbots.gemini_chatbot import gemini_answer_query
from chatbots.llama_3b_instruct_chatbot import llama_answer_query

# Configure environment for CUDA and Streamlit
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# allow nested event loops
import nest_asyncio
nest_asyncio.apply()

from pathlib import Path

# Ensure data directories exist
DATA_DIR = Path(__file__).parent
data_pdf_dir = DATA_DIR / "data/pdf"
data_csv_dir = DATA_DIR / "data/csv"
data_pdf_dir.mkdir(parents=True, exist_ok=True)
data_csv_dir.mkdir(parents=True, exist_ok=True)

# Helper: send & clear
def submit_message():
    user_input = st.session_state.user_input.strip()
    if not user_input:
        return
    # Append user message
    st.session_state.chat_history.append(("user", user_input))
    # Choose chatbot
    if st.session_state.bot_choice == "Gemini":
        try:  
            ai_response = _loop.run_until_complete(
                gemini_answer_query(user_input)
            )        
        except Exception as e:
            st.error(f"Error: {e}")
            ai_response = "An error occurred while processing your request. Try again later."
    else:
        try:
            ai_response = _loop.run_until_complete(
                llama_answer_query(user_input)
            )
        except Exception as e:
            st.error(f"Error: {e}")
            ai_response = "An error occurred while processing your request. Try again later."
    # Append AI response
    st.session_state.chat_history.append(("ai", ai_response))
    st.session_state.user_input = ""

# Main app
def main():
    # Apply purple-pink gradient background
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #2A7B9B, #17124A);
            background-attachment: fixed;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Header styles
    st.markdown(
        """
        <style>
        .header {
            text-align: center;
            font-size: 2rem;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True
    )

    import base64

    img_path = Path(__file__).parent / "images" / "sophos.png"
    encoded = base64.b64encode(img_path.read_bytes()).decode()
    img_html = f'<img src="data:image/png;base64,{encoded}" width="100" />'

    # Header with image instead of emoji and align center
    st.markdown(
        f"""
        <div class="header">
            {img_html}
            <h1>Sophos (Σοφός)</h1>
            <p>A chatbot that answers questions based on uploaded files and querying the internet.</p>
        </div>
        """, unsafe_allow_html=True
    )

    # Sidebar: chatbot selection
    st.sidebar.title("Choose your chatbot")
    bot_option = st.sidebar.radio(
        label="Select a bot", options=["Google Gemini", "Finetuned Llama-3B-Instruct"], key="bot_choice_radio"
    )
    st.session_state.bot_choice = "Gemini" if bot_option == "Google Gemini" else "Llama"

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload a PDF or CSV file", type=["pdf", "csv"], key="file_uploader"
    )
    if uploaded_file:
        filename = uploaded_file.name
        target_dir = data_pdf_dir if filename.lower().endswith(".pdf") else data_csv_dir
        target = target_dir / filename
        with open(target, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"Saved '{filename}' to '{target.parent}/'")

    # Clear history button
    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.session_state.user_input = ""

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "bot_choice" not in st.session_state:
        st.session_state.bot_choice = "Gemini"
        st.session_state.bot_choice_radio = "Google Gemini"

    # Chat input and send button
    st.text_input(
        "Type your message here", key="user_input", placeholder="Enter text and click Send"
    )
    st.button("Send", on_click=submit_message)

    # Display chat history
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    # Footer
    st.markdown(
        """
        <hr style='margin-top: 2rem;' />
        <div style='text-align: center; font-size: 0.9rem;'>
            Created by Mazen Soliman
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()