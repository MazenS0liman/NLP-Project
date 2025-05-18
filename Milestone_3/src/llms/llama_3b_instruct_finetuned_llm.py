import os
import sys
import streamlit as st
from init import torch, transformers
from peft import PeftModel

os.environ["HF_HUB_OFFLINE"] = "1"

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),  # .../Milestone_3/src/llms
        "..",                        # up to .../Milestone_3/src
        "models",                    # into .../Milestone_3/src/models
        "Llama-3.2-3B-Instruct"
    )
)
ADAPTER_DIR = os.path.join(BASE_DIR, "checkpoint-question-answering-task")
DEVICE = torch.device(f"cuda:{0 if torch.cuda.is_available() else 1}") 

print("CWD:", os.getcwd())
print("BASE_DIR exists?", os.path.isdir(BASE_DIR), "→", BASE_DIR)
print("Absolute path:", os.path.abspath(ADAPTER_DIR))

# Check if the model directory exists
if not os.path.isdir(BASE_DIR):
    print(f"❌ BASE_DIR not found: {BASE_DIR}", file=sys.stderr)
    sys.exit(1)

torch_dtype = torch.bfloat16
attn_implementation = "eager"

@st.cache_resource(
        show_spinner="Loading model…",
        ttl=3600,  # Cache for 1 hour
)
def load_model():
    # 1) Tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        BASE_DIR, 
        local_files_only=True, 
        trust_remote_code=True
    )
    tokenizer.pad_token_id = 0
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # 2) QLoRA & 4-bit base
    bnb = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage="uint8",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    base = transformers.AutoModelForCausalLM.from_pretrained(
        BASE_DIR,
        quantization_config=bnb,
        device_map="balanced",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True,
    ).to(DEVICE)
    base.generation_config.pad_token_id = tokenizer.eos_token_id

    # 3) PEFT adapter
    model = PeftModel.from_pretrained(
        base, 
        ADAPTER_DIR, 
        local_files_only=True
    )
    model.eval().to(DEVICE)

    return tokenizer, model

# —————————————————————————————————————————————————————————————
# 5) Summarization helper (same as before)
# —————————————————————————————————————————————————————————————
DEFAULT_SYSTEM_PROMPT = """
Given the user input below, generate a concise answer that:
- Captures the main ideas and essential details related to the user question.
- Avoid including any information not present in the context.
- Only provide the answer without any additional commentary or explanation or justification.
""".strip()

def generate_prompt(context: str, question: str, system_prompt: str = DEFAULT_SYSTEM_PROMPT) -> str:
    return f"""
    ### System: 
    You are an expert QA assistant that answers user's question based on the context.

    ### Instruction: {system_prompt}

    ### Context: {context}

    ### Input:
    {question}

    ### Response:
    """

def generate_response(model, tokenizer, context: str, question: str, system_prompt:str = None) -> str:
    if system_prompt is None:
        prompt = generate_prompt(context, question)
    else:
        prompt = generate_prompt(context, question, system_prompt)

    encoding = tokenizer(prompt, return_tensors="pt")

    # move each tensor to the same device as your model
    device = next(model.parameters()).device
    inputs = {k: t.to(device) for k, t in encoding.items()}
    inputs_length = inputs["input_ids"].size(1)


    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=128, temperature=0.7)
    out = tokenizer.decode(outputs[0][inputs_length:], skip_special_tokens=True)

    return out