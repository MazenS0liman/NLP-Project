#!/usr/bin/env python
# coding: utf-8

import os
import warnings
warnings.filterwarnings('ignore')

import asyncio
import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX')
os.environ['TESSERACT'] = os.getenv('TESSERACT_CMD')
os.environ['USER_AGENT'] = "myagent"

from typing import List, Any

# LangChain components
from langchain.prompts import PromptTemplate
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tracers.context import tracing_v2_enabled

# Custom agents
from agents.query_agent import generate_queries
from agents.summary_agent import generate_summary

# Tools
from tools.browse_web import browse_web
from tools.parse_csv import retrieve_csv_data
from tools.parse_pdfs import retrieve_pdf_content

# Workflow
from workflow import retrieve_data

from grpc.experimental.aio import init_grpc_aio
# init_grpc_aio()

from dotenv import load_dotenv
load_dotenv()

# Model & API keys
MODEL_NAME = "gemini-1.5-flash-8b"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT")

def truncate_preview(obj: Any, max_len: int = 200) -> str:
    s = repr(obj)
    return s if len(s) <= max_len else s[:max_len] + "..."

# ReAct-style Prompt: recursive until final answer
PROMPT = PromptTemplate.from_template("""
System: You are an expert QA assistant. Use tools and ReAct reasoning to find the best answer.
You may call tools multiple times in sequence until you can confidently respond.
Provide resources and citations for your answers if possible.
                                      
# Instructions:
- Analyse the provided context and the user's question.
- If the question is ambiguous or complex, generate clarifying questions using Generate_Questions tool.
- From the generated questions, select the most relevant one to ask the user or use Browse_Web, Parse_PDF, or Parse_CSV tools to gather more information.

# What are the inputs available for each tool:
- Generate_Questions: generate_queries(input: str)
    -- input: your question or context to generate clarifying questions.
- Browse_Web: browse_web(query: str)
    -- query: your question or context to search the web.
- Summarize_Text: generate_summary(context: str)
    -- context: the text you want to summarize.
- Parse_PDF: retrieve_pdf_content(query: str)
    -- query: your question or context to extract text from PDF files.
- Parse_CSV: retrieve_csv_data(query: str)
    -- query: your question or context to extract text from CSV files.

Available Tools: {tools}
When you decide on an action, respond exactly:
Thought: <…>
Action: <tool name>  # must be one of [{tool_names}]
Action Input: <input for the tool>

After observing the result, continue:
Thought: <new reasoning>
Either call another tool or return:
Final Answer: <your answer>
And do NOT propose any more Thoughts or Actions.

# Examples:
### Example 1:
1. User: What is the capital of France?
2. Thought: I need to gather information about France's capital..
3. Action: Use the Browse_Web tool to search for the capital of France.
4. Action Input: "What is the capital of France?"
5. Thought: I have gathered the necessary information to answer the user's question.
6. Final Answer: The capital of France is Paris.

### Example 2:
1. User: What is the id of Mr. Mohamed from our csv data?
2. Thought: I need to check the CSV data for Mr. Mohamed's ID.
3. Action: Use the Parse_CSV tool to search for Mr. Mohamed's ID.
4. Action Input: "Mr. Mohamed ID in the CSV data"
5. Thought: I have found the ID for Mr. Mohamed in the CSV data.
6. Final Answer: The ID of Mr. Mohamed is 12345.

Context: {context}

User Question: {input}
{agent_scratchpad}
"""
)

tools = [
    Tool(
        name="Generate_Questions",
        func=generate_queries,
        description="Produce clarifying questions for ambiguous or complex queries."
    ),
    Tool(
        name="Browse_Web",
        func=browse_web,
        description="Search the web for information."
    ),
    Tool(
        name="Summarize_Text",
        func=generate_summary,
        description="Summarize provided text input."
    ),
    Tool(
        name="Parse_PDF",
        func=retrieve_pdf_content,
        description="Extract text from PDF files."
    ),
    Tool(
        name="Parse_CSV",
        func=retrieve_csv_data,
        description="Extract text from CSV files."
    )
]

# Memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",            
    output_key="output"
)

# Initialize LLM Agent
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    convert_system_message_to_human=True,
    verbose=True,
)
agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=PROMPT,
)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    memory=memory,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

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
async def gemini_answer_query(
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

    with tracing_v2_enabled(project_name=LANGCHAIN_PROJECT):
        result: dict = await agent_executor.acall({
            "input":   query,
            "context": context,
        })
        final_answer = result["output"]
        if not final_answer:
            return "I do not know the answer!"
        
        print(bcolors.OKGREEN + "Final Answer:" + bcolors.ENDC, final_answer)
        return final_answer

