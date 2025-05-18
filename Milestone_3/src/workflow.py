import os
os.environ['USER_AGENT'] = 'myagent'

from typing import TypedDict, List, Any

# LangGraph
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Agents
from agents import browse_agent, summary_agent

# Tools
from tools.browse_web import browse_web
from tools.parse_csv import retrieve_csv_data
from tools.parse_pdfs import retrieve_pdf_content

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

MAX_PREVIEW = 100

def preview(obj: Any, max_len: int = MAX_PREVIEW) -> str:
    s = repr(obj)
    if len(s) > max_len:
        return s[:max_len] + "...⏎(truncated)"
    return s

# ——————————————————————————————————————————————————————————————
# Typed state for our graph
class AgentState(TypedDict):
    input: str
    queries: List[str]
    search_results: List[dict]
    summary: str
    retrieved_pdf_context: List[str]

# ——————————————————————————————————————————————————————————————
def surf_web(state: AgentState) -> List[dict]:
    print(bcolors.OKCYAN + "Searching the web..." + bcolors.ENDC)
    print(
        bcolors.OKCYAN + "Current State:" + bcolors.ENDC,
        preview(state)
    )
    print("---" * 20)

    try:
        query = state["input"]

        # snippets = await browse_agent.get_information(query, num=5)
        snippets = []
        if snippets == []:
            snippets = browse_web(query, num=10)
        return {"search_results": snippets}
    except Exception as e:
        print(bcolors.FAIL + "Error searching the web:" + bcolors.ENDC, e)
        return {"search_results": []}

async def search_queries(state: AgentState):
    print(bcolors.OKCYAN + "Searching the web..." + bcolors.ENDC)
    print(
        bcolors.OKCYAN + "Current State:" + bcolors.ENDC,
        preview(state)
    )
    print("---" * 20)

    try:
        queries = state['queries']
        out = []
        for q in queries:
            snippets = await browse_web(q, num=5)
            if snippets == []:
                snippets = browse_agent.get_information(q, num=5)

            out.append({"query": q, "snippets": snippets})

        return {"search_results": out}
    except Exception as e:
        print(bcolors.FAIL + "Error searching for queries:" + bcolors.ENDC, e)
        return {"search_results": []}

def summarize_batch(state: AgentState) -> List[dict]:
    print(bcolors.OKCYAN + "Summarizing results..." + bcolors.ENDC)
    print(
        bcolors.OKCYAN + "Current State:" + bcolors.ENDC,
        preview(state)
    )
    print("---" * 20)

    try:
        summaries = []
        for entry in state["search_results"]:
            summary = summary_agent.generate_summary(entry, use_groq=True)
            if not summary:
                summary = "No relevant information found."
            elif isinstance(summary, str):
                # If it’s a string, we’re good
                pass
            elif hasattr(summary, "content"):
                # If it’s a message object, get the content
                summary = summary.content
            else:
                # Otherwise, raise an error
                raise ValueError(f"Unexpected type for summary: {type(summary)}")
            
            summaries.append(summary)

        return {"summary": '\n'.join(summaries)}
    
    except Exception as e:
        print(bcolors.FAIL + "Error summarizing results:" + bcolors.ENDC, e)
        return {"summary": "Error summarizing results."}

def retrieve_context_from_pdfs(state: AgentState) -> List[dict]:
    '''
    This function is not used in the current workflow but can be used to retrieve answers from the summaries.
    '''
    print(bcolors.OKCYAN + "Retrieving context from pdfs..." + bcolors.ENDC)
    print(
        bcolors.OKCYAN + "Current State:" + bcolors.ENDC,
        preview(state)
    )
    print("---" * 20)

    try:    
        query = state["input"]
        retrieved_pdf_context = retrieve_pdf_content(pdfs_path="./data/pdf/", query=query)
        return {"retrieved_pdf_context": retrieved_pdf_context}
    except Exception as e:
        print(bcolors.FAIL + "Error retrieving context from pdfs:" + bcolors.ENDC, e)
        return {"retrieved_pdf_context": []}

def retrieve_context_from_csv(state: AgentState) -> str:
    '''
    Retrieve data from the database based on the query.
    '''
    print(bcolors.OKCYAN + "Retrieving data from csv files..." + bcolors.ENDC)
    print(
        bcolors.OKCYAN + "Current State:" + bcolors.ENDC,
        preview(state)
    )
    print("---" * 20)

    try:    
        query = state["input"]
        retrieved_data = retrieve_csv_data(csv_folder_path="./data/csv/", query=query)
        return {"retrieved_csv_data": retrieved_data}
    except Exception as e:
        print(bcolors.FAIL + "Error retrieving data from csv files:" + bcolors.ENDC, e)
        return {"retrieved_csv_data": []}

# ——————————————————————————————————————————————————————————————
# Build the graph
workflow = StateGraph(AgentState)

workflow.add_node(
    "surf_web",
    surf_web,
)
workflow.add_node(
    "summarize_information_from_web",
    summarize_batch,
)
workflow.add_node(
    "retrieve_context_from_pdfs",
    retrieve_context_from_pdfs,
)
workflow.add_node(
    "retrieve_context_from_csv",
    retrieve_context_from_csv,
)

# Entry and finish points
workflow.set_entry_point("surf_web")
workflow.set_finish_point("retrieve_context_from_csv")

# Edges
workflow.add_edge("surf_web", "summarize_information_from_web")
workflow.add_edge("summarize_information_from_web", "retrieve_context_from_pdfs")
workflow.add_edge("retrieve_context_from_pdfs", "retrieve_context_from_csv")

# Memory
memory = MemorySaver()

# Compile graph
graph = workflow.compile(checkpointer=memory)

# Specify an ID for the thread
config = {"configurable": {"thread_id": "abc123"}}

def retrieve_data(query: str) -> str:
    '''
    Main entry point: break down the input into queries, search, then summarize.
    Returns a formatted string of all query–summary pairs.
    '''
    # Initial state
    state: dict = {"input": query}

    for step in graph.stream(
        state,
        stream_mode="values",
        config=config,
    ):
        update = step
        print(
            bcolors.OKBLUE + "Step result:" + bcolors.ENDC,
            preview(update)
        )
        state.update(update)

    print(
        bcolors.OKGREEN + "Final State:" + bcolors.ENDC,
        preview(state)
    )
    return [{"output": f"""
    ---- Info ---
    #### Input:
    {state['input']}

    #### Information from the Internet:
    {state.get("summary", "​No summary generated.")}

    #### Information from the PDFs:   
    {chr(10).join(state.get("retrieved_pdf_context", []))}

    ### Information from the CSV:
    {chr(10).join(state.get("retrieved_csv_data", []))}
    ---- End ----
    """.strip()}, {"retrieved_pdf_context": state.get("retrieved_pdf_context", [])}, {"retrieved_csv_data": state.get("retrieved_csv_data")}, {"summary": state.get("summary", "​No summary generated.")}]

if __name__ == "__main__":
    question = "Who is the president of the USA at 2025?"
    # result = asyncio.run(chat(question))
    out = retrieve_data(question)
    print(bcolors.OKGREEN + "Output:\n" + bcolors.ENDC, out)
