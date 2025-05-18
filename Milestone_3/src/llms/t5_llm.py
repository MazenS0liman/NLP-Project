import os
from init import torch, transformers
from langchain_huggingface import HuggingFacePipeline

MODEL_NAME = "BeIR/query-gen-msmarco-t5-base-v1"
MODEL_DIR = "./models/T5-Base-V1"
MAX_LENGTH = 512
DEVICE = 0

# —————————————————————————————————————————————————————————————
# 2. Model save / load utilities
# —————————————————————————————————————————————————————————————
def save_model(model, tokenizer, save_dir: str = MODEL_DIR):
    """Save HF model and tokenizer to disk."""
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"✅ Model and tokenizer saved in {save_dir}")

def load_model(save_dir: str = MODEL_DIR):
    """Load HF model and tokenizer from disk."""
    model = transformers.T5ForConditionalGeneration.from_pretrained(
        save_dir,
    ).to(device=DEVICE)
    tokenizer = transformers.T5Tokenizer.from_pretrained(
        save_dir
    ).to(device=DEVICE)
    return model, tokenizer

# —————————————————————————————————————————————————————————————
# 3. (Re)build the pipeline + LangChain LLM
# —————————————————————————————————————————————————————————————
def build_llm(model=None, tokenizer=None, save_dir: str = MODEL_DIR, model_name: str = MODEL_NAME):
    """
    Returns a HuggingFacePipeline-wrapped LLM.
    If model/tokenizer are None, it tries to load from disk; otherwise
    fallbacks to downloading from Hugging Face hub.
    """
    if model is None or tokenizer is None:
        try:
            model, tokenizer = load_model(save_dir)
        except Exception:
            # Fallback: fetch from hub if not saved yet
            print("⚠️ Saved model not found. Downloading from HF hub...")
            print(f"📦 Downloading {model_name}...")
            model = transformers.T5ForConditionalGeneration.from_pretrained(model_name)
            tokenizer = transformers.T5Tokenizer.from_pretrained(model_name)
            save_model(model, tokenizer, save_dir)

    text_gen_pipeline = transformers.pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    return HuggingFacePipeline(
        pipeline=text_gen_pipeline,
        model_kwargs={"max_length": MAX_LENGTH}
    )

# Instantiate once at module load
t5_llm = build_llm()