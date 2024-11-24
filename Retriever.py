import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
print("SQLite version used by modified Python:", sqlite3.sqlite_version)

#imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
#import transformers
#import torch
#from langchain_ollama import OllamaEmbeddings
import chromadb
#from langchain_community import embeddings
#from langchain_community.embeddings import OllamaEmbeddings
import chromadb.utils.embedding_functions as embedding_functions
#from langchain_huggingface import HuggingFaceEmbeddings
import ollama
from tqdm import tqdm

from Embedding import OllamaEmbeddingsWrapper

urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

#embeddings = OllamaEmbeddings(model="llama3")

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)

for doc in docs_list:
    print(doc)
doc_splits = text_splitter.split_documents(docs_list)


#print(f"Split documents into {len(doc_splits)} chunks.")

chroma_client = chromadb.PersistentClient(path="/workspaces/Self_Reflective_Rag/db")

#ollama_ef = embedding_functions.OllamaEmbeddingFunction(
#    url="http://localhost:11434/api/embeddings",
#    model_name="nomic-embed-text",
#)

#client = chromadb.Client()
#collection =  client.create_collection(name="rag-chroma")

#client = chromadb.PersistentClient(path="ollama")


#embedding_function = OllamaEmbeddingsWrapper(model='nomic-embed-text')

#vectorstore = Chroma.from_documents(
#    client=chroma_client,
#    documents=doc_splits,
#    collection_name="rag-chroma",
#    embedding=embedding_function,
#)

#embeddings = HuggingFaceEmbeddings("nomic-ai/nomic-embed-text-v1", trust_remote_code=True)
# Add to vectorDB

embedding_wrapper = OllamaEmbeddingsWrapper()
total_docs = len(doc_splits)

with tqdm(total=total_docs, desc="Processing documents") as pbar:
    vectorstore = Chroma.from_documents(
        client=chroma_client,
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=embedding_wrapper,
    )
    pbar.update(total_docs)

print("Vectorstore creation complete.")


retriever = vectorstore.as_retriever()

print(f"Total documents in vectorstore: {vectorstore._collection.count()}")

sample_query = "What is an AI agent?"
results = retriever.get_relevant_documents(sample_query)
print(f"Retrieved {len(results)} documents for sample query: '{sample_query}'")

