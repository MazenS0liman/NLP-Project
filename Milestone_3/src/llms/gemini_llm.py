import os
from langchain_google_genai import ChatGoogleGenerativeAI

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# —————————————————————————————————————————————————————————————
# 1. Initialize the LLM
# —————————————————————————————————————————————————————————————
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.7,
    max_tokens=150,
    convert_system_message_to_human=True,
    verbose=True,
)
