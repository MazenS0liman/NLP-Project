import os
import dotenv
import pandas as pd
from sqlalchemy import create_engine

from langchain_groq import ChatGroq
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent

dotenv.load_dotenv()

# Create Gemini-powered SQL agent
from dotenv import load_dotenv
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

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
# Define the LLM model
llm = ChatGroq(temperature=0.7, model_name=MODEL_NAME, api_key=GROQ_API_KEY)

def retrieve_csv_data(query: str, csv_folder_path: str = "./data/csv"):
    """Retrieve data from the database based on the query."""
    try:
        # Load CSVs and create database engine
        csv_files_path = [os.path.join(csv_folder_path, f) for f in os.listdir(csv_folder_path) if f.endswith('.csv')]
        database_url = os.getenv("DATABASE_URL", "sqlite:///mydata.db")
        engine = create_engine(database_url)

        # Load each CSV into a table named after the file (without extension)
        for path in csv_files_path:
            table_name = os.path.splitext(os.path.basename(path))[0]
            df = pd.read_csv(path)
            df.to_sql(table_name, engine, index=False, if_exists="replace")

        # Initialize SQLDatabase
        db = SQLDatabase(engine=engine)
        print("Dialect:", db.dialect)
        print("Tables:", db.get_usable_table_names())

        agent_executor = create_sql_agent(
            llm=llm,
            db=db,
            verbose=True,
            agent_type="zero-shot-react-description",
            max_iterations=3,
            handle_parsing_errors="output nothing",
            request_timeout=600,
        )

        # Use the agent to execute the query
        response = agent_executor.invoke({"input": query})
        return response
    except Exception as e:
        print(bcolors.FAIL + "Error retrieving CSV data:" + bcolors.ENDC, e)
        return ""
    