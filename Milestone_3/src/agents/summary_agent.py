import os
import time
import textwrap
from dotenv import load_dotenv
from groq import Groq
# from llms.llama_3b_instruct_finetuned_llm import summarize as local_summarize

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

MODEL_NAME = "llama3-70b-8192"
SYSTEM_MESSAGE = """
You are a concise summarization assistant. Use only the provided information.
Do not inject any external knowledge.
"""

# Approximate token limit per chunk (Groq on-demand: 6000 TPM)
# Here we target ~2000 tokens ≈ 8000 characters.
MAX_CHARS_PER_CHUNK = 8000
MAX_RETRIES = 3
BACKOFF_BASE   = 1.0  # seconds

def call_groq(messages):
    """Invoke Groq with retries on 413, raising on other errors."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=128,
                temperature=0.7,
                top_p=0.9,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            # If it's a 413 rate-limit, retry after backoff
            if "413" in err and attempt < MAX_RETRIES:
                wait = BACKOFF_BASE * (2 ** (attempt - 1))
                print(f"[Groq 413] retry {attempt}/{MAX_RETRIES} after {wait:.1f}s...")
                time.sleep(wait)
                continue
            # For other errors or last retry, re-raise
            raise

def generate_summary(context: str, use_groq: bool = True) -> str:
    """Summarize `context`, chunking and falling back as needed."""
    if not context.strip():
        return ""
    
    # Split into overlapping chunks of ~MAX_CHARS_PER_CHUNK
    raw_chunks = textwrap.wrap(context, MAX_CHARS_PER_CHUNK, break_long_words=False, replace_whitespace=False)
    summaries = []
    
    for i, chunk in enumerate(raw_chunks, start=1):
        print(f"Summarizing chunk {i}/{len(raw_chunks)} (≈{len(chunk)} chars)...")
        if use_groq:
            try:
                msg = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user",   "content": chunk},
                ]
                summary = call_groq(msg)
            except Exception as groq_err:
                print(f"Groq failed: {groq_err}\nFalling back to local summarizer.")
                # summary = local_summarize(text=chunk).strip()
                summary = "[LOCAL_SUMMARY_PLACEHOLDER]"  # replace with your local call
        else:
            # summary = local_summarize(text=chunk).strip()
            summary = "[LOCAL_SUMMARY_PLACEHOLDER]"
        
        summaries.append(summary)
    
    # If multiple chunk-summaries, stitch them and optionally run a final pass
    if len(summaries) > 1:
        combined = "\n\n".join(summaries)
        print("Combining chunk summaries into final summary…")
        if use_groq:
            try:
                final_msg = [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user",   "content": combined},
                ]
                return call_groq(final_msg)
            except Exception:
                # return local_summarize(text=combined).strip()
                return combined  # or your local fallback
        else:
            return combined
    
    return summaries[0]
