#!/usr/bin/env python
# coding: utf-8

import re
import os
import warnings
warnings.filterwarnings('ignore')

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import PyPDF2

import pytesseract
pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_CMD')
os.environ['TESSDATA_PREFIX'] = os.getenv('TESSDATA_PREFIX')
os.environ['TESSERACT'] = os.getenv('TESSERACT_CMD')

from io import StringIO 
from lxml import etree
from pathlib import Path
from typing import List
from functools import lru_cache

from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
from unstructured.staging.base import dict_to_elements
from unstructured.documents.elements import Table, Title, Image, Header

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.embeddings.huggingface import HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts.prompt import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain

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
def extract_pdf_elements(pdf_path: str, hi_res_model_name: str = "yolox") -> list:
    """Partition PDF into elements."""
    return partition_pdf(
        filename=pdf_path,
        extract_images_in_pdf=False,
        infer_table_structure=False,
        strategy="hi_res",
        hi_res_model_name=hi_res_model_name,
        max_characters=3000,
        combine_text_under_n_chars=100
    )

def filter_elements(elements: list) -> list:
    """Remove all Table elements and strip off anything from the References/Bibliography 
    section (by title) onward.
    """
    filtered = []
    stop = False

    for el in elements:
        # 1) if it’s a Table, skip it
        if isinstance(el, Table) or isinstance(el, Image) or isinstance(el, Header):
            continue

        # 2) if it’s a Title that says “References” or “Bibliography”, stop processing
        if isinstance(el, Title) and re.match(r"^(References|Bibliography)", el.text or "", re.I):
            stop = True
            break

        # otherwise keep it
        filtered.append(el)

    return filtered

def chunk_text(elements: list) -> list:
    """Split extracted text elements into manageable chunks."""
    sections = []
    current_title = None
    current_text = []

    for element in elements:
        if element.__class__.__name__ == "Title":
            if current_title is not None:
                sections.append((current_title, "\n".join(current_text)))
            current_title = element.text.strip()
            current_text = []
        else:
            txt = getattr(element, "text", None)
            if txt:
                current_text.append(txt.strip())

    if current_title is not None:
        sections.append((current_title, "\n".join(current_text)))

    docs = [
        Document(
            page_content=text, 
            metadata={"title": title}
        )
        for title, text in sections
    ]

    nonempty_docs = [d for d in docs if d.page_content and d.page_content.strip()]

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=10,
        length_function=len,
        separator=" "
    )

    split_docs = splitter.split_documents(nonempty_docs)

    return split_docs

@lru_cache(maxsize=1)
def get_fastembed_model():
    """Loads the FastEmbedEmbeddings model once and reuses it."""
    return FastEmbedEmbeddings()

def create_embeddings():
    """Return a singleton FastEmbedEmbeddings instance."""
    return get_fastembed_model()

def build_vectorstore(docs: list, embeddings) -> Qdrant:
    """Build or load a Qdrant vectorstore."""
    vectorstore = Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",          # ← entirely in-process
        collection_name="rag",
    )
    return vectorstore

def create_retriever(vectorstore: Qdrant, k: int = 10) -> Qdrant:
    """Create a retriever from vectorstore."""
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": k,
        },
    )

def answer_query(retriever: Qdrant, query: str) -> dict:
    """Run the QA chain against a query."""
    template = """
    You are a smart assistant helping to find the best section(s) in a document to answer a user's question.

    Here are the available section titles:
    {section_titles}

    Given the question: "{question}"

    Return a comma-separated list of the most relevant section titles to search for an answer.
    """

    prompt = PromptTemplate(template=template, input_variables=["question", "context"])

    llm = ChatGroq(temperature=0.7, model_name=MODEL_NAME, api_key=GROQ_API_KEY)

    doc_chain = load_qa_with_sources_chain(llm, chain_type="map_reduce")
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    qa_chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator_chain,
        combine_docs_chain=doc_chain,
    )
    
    return qa_chain.run({
        "question": query,
        "section_titles": [doc.metadata["title"] for doc in retriever.get_relevant_documents(query)],
        "chat_history": [],
    })


def parse_pdf(pdfs_path: str, query: str) -> Qdrant:
    """Parse all pdfs in a directory and create a retriever."""
    elements = extract_pdf_elements(pdfs_path)
    docs = chunk_text(elements)
    
    embeddings = create_embeddings()
    vectorstore = build_vectorstore(docs, embeddings)
    retriever = create_retriever(vectorstore)
    return retriever

def retrieve_pdf_content(query: str, pdfs_path: str = "./data/pdf", k: int = 5) -> List[str]:
    """
    Parse all PDFs in `pdfs_dir`, build one combined vectorstore,
    and return the top-k page_content strings relevant to `query`.
    """
    try:
        pdf_folder = Path(pdfs_path)
        if not pdf_folder.is_dir():
            raise FileNotFoundError(f"PDF directory not found: {pdfs_path}")
        
        # 1) Partition & chunk every PDF
        all_docs = []
        for pdf_path in pdf_folder.glob("*.pdf"):
            try:
                # 1) partition
                els = extract_pdf_elements(str(pdf_path))
                # 2) filter out tables & references
                els = filter_elements(els)
                # 3) chunk remaining text
                chunks = chunk_text(els)
                all_docs.extend(chunks)
            except Exception as e:
                print( bcolors.WARNING + "Warning" + bcolors.ENDC + ": skipping {pdf_path} page due to image error: {e}")
                doc = Document(page_content="", metadata={"title": pdf_path.stem})
                for page_num in range(len(PyPDF2.PdfReader(pdf_path).pages)):
                    text = PyPDF2.PdfReader(pdf_path).pages[page_num].extract_text()
                    if text:
                        doc.page_content += text
                if doc.page_content.strip():
                    all_docs.append(doc)

        if not all_docs:
            return []

        # 2) Build your embedding store once over all chunks
        embeddings  = create_embeddings()
        vectorstore = build_vectorstore(all_docs, embeddings)
        retriever   = create_retriever(vectorstore, k=k)

        # 3) Get the top-k relevant chunks
        retrieved_docs = retriever.get_relevant_documents(query)
        return [doc.page_content for doc in retrieved_docs]
    except Exception as e:
        print(bcolors.FAIL + "Error" + bcolors.ENDC + f" retrieving PDF content: {e}")
        return []

def main():
    """Main function to run the script."""
    pdf_path = "path/to/your/pdf.pdf"  # Replace with your PDF path
    query = "What is the main topic of the document?"  # Replace with your query

    retriever = parse_pdf(pdf_path, query)
    answer = answer_query(retriever, query)

    print("Answer:", answer)
