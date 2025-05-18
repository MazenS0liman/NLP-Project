import re
import json
from langchain.prompts import PromptTemplate
from langchain_core.runnables.base import RunnableSequence
# from llms.t5_llm import t5_llm  # previously used
from llms.gemini_llm import gemini_llm

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

# —————————————————————————————————————————————————————————————
# 1. System prompt
# —————————————————————————————————————————————————————————————
agent_prompt = PromptTemplate.from_template("""
### Role:
You are an expert educational content creator specializing in generating insightful, varied, and context-aware questions.

### Instruction:
Based solely on the input text, produce **five** distinct questions that:
- Cover factual details, inference, and critical thinking
- Vary in format (e.g., multiple-choice, open-ended, true/false)
- Use clear, concise language
- Do **not** introduce any information not present in the input
- Produce **five** JSON-formatted questions that are relevant to the input text.

### Output format:
Return **only** a JSON array of strings, e.g.:
[
  {{"Q1": "What is ...?"}},
  {{"Q2": "Which ...?"}},
  {{"Q3": "Does/Do ...?"}},
  {{"Q4": "What is ...?"}},
  {{"Q5": "Why ...?"}}
]

### IMPORTANT:
Your _only_ output must be a JSON array of strings.  
Do not include any markdown fences, bullet points, explanation, or extra text.
Do not include any newlines or spaces outside the JSON array.
Do not include any text that is not part of the JSON array.

### Examples:
Input:  
---BEGIN SOURCE---  
Photosynthesis allows plants to convert sunlight into chemical energy stored as glucose.  
---END SOURCE---

Response:
[
{{"Q1": "What molecule do plants produce during photosynthesis?"}},
{{"Q3": "Which process do plants use to convert sunlight into energy?"}},
{{"Q2": "Does: photosynthesis occur in animals?"}},
{{"Q4": "What is the primary purpose of photosynthesis?"}},
{{"Q5": "Why is photosynthesis important for plants?"}}
]

### Input:
---BEGIN SOURCE---
{input}
---END SOURCE---
                                            
### Response:
""")

# —————————————————————————————————————————————————————————————
# 2. Build the prompt‐only chain
# —————————————————————————————————————————————————————————————
agent_runnable: RunnableSequence = agent_prompt | gemini_llm

# —————————————————————————————————————————————————————————————
# 3. Exposed function for your UI
# —————————————————————————————————————————————————————————————
def generate_query(user_text: str) -> str:
    """
    Runs the prompt on the user’s input and returns a query.
    """
    # If using memory, pass input_key matching your template (here: "input")
    return agent_runnable.invoke({"input": user_text})

def generate_queries(input: str):
    try:
        raw_msg = generate_query(input)
        raw_text = raw_msg.content if hasattr(raw_msg, "content") else str(raw_msg)

        # Attempt to extract JSON-looking substring
        m = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if m:
            gen_queries = json.loads(m.group(0))

            # 3) If it’s a list of single-key dicts, flatten to just the strings:
            if isinstance(gen_queries, list) and gen_queries and all(isinstance(item, dict) for item in gen_queries):
                # e.g. [{"Q1": "…”}, {"Q2": "…”}, …]
                gen_queries = [next(iter(d.values())) for d in gen_queries]

            # 4) Now it’s guaranteed to be List[str]
            if not all(isinstance(q, str) for q in gen_queries):
                raise ValueError(f"Expected List[str], got {gen_queries!r}")

            return gen_queries
    except Exception as e:
        print(bcolors.FAIL + "Error generating queries:" + bcolors.ENDC, e)
        return []