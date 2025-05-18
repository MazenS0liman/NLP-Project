#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path
from functools import lru_cache

from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant

# ——————————————————————————————————————————————————————————————
@lru_cache(maxsize=1)
def get_fastembed_model():
    return FastEmbedEmbeddings()

def create_embeddings():
    return get_fastembed_model()

def load_text_files(txt_dir: str) -> list[Document]:
    docs = []
    for txt_path in Path(txt_dir).glob("*.txt"):
        try:
            text = txt_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = txt_path.read_text(encoding="latin-1")
        if not text.strip():
            continue
        docs.append(Document(
            page_content=text,
            metadata={"source": txt_path.name}
        ))
    return docs

def chunk_documents(docs: list[Document], chunk_size: int = 500, overlap: int = 50) -> list[Document]:
    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separator="\n"
    )
    return splitter.split_documents(docs)

def build_qdrant_index(docs: list[Document], embeddings) -> Qdrant:
    return Qdrant.from_documents(
        documents=docs,
        embedding=embeddings,
        location=":memory:",
        collection_name="text_chunks",
    )

def parse_txt(txt_dir: str = "./data/texts"):
    raw_docs = load_text_files(txt_dir)
    print(f"Loaded {len(raw_docs)} text files.")

    chunked_docs = chunk_documents(raw_docs, chunk_size=1000, overlap=100)
    print(f"Split into {len(chunked_docs)} chunks.")

    embeddings = create_embeddings()
    vectorstore = build_qdrant_index(chunked_docs, embeddings)
    print("Qdrant index created with collection:", vectorstore.collection_name)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )

    return retriever
