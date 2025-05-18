from workflow import retrieve_data
from llms.llama_3b_instruct_finetuned_llm import load_model, generate_response

# Load the model and tokenizer
tokenizer, model = load_model()

# ——————————————————————————————————————————————————————————————
# This is a simple class to add colors to the output in the terminal.
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RED = '\033[31m'

# ——————————————————————————————————————————————————————————————
# Main query interface
async def llama_answer_query(
    query: str,
    context: str = None,
) -> str:
    if not query.strip():
        return ""
    
    if context is None:
        fetched_data = retrieve_data(query=query)
        web_ctx = "\n".join(f"- {s}" for s in fetched_data[2].get("summary", []))
        pdf_ctx = "\n".join(f"- {p}" for p in fetched_data[1].get("retrieved_pdf_context", []))
        csv_ctx = "\n".join(f"- {c}" for c in fetched_data[0].get("retrieved_csv_data", []))
        
        context = "\n".join([
            "Web context:",
            web_ctx or "(none)",
            "",
            "PDF context:",
            pdf_ctx or "(none)",
            "CSV context:",
            csv_ctx or "(none)",
        ])

    final_answer: dict = generate_response(
        model=model,
        tokenizer=tokenizer,
        context=context,
        question=query,
    )
    if not final_answer:
        return "I do not know the answer!"
    
    print(bcolors.OKGREEN + "Final Answer:" + bcolors.ENDC, final_answer)
    return final_answer

