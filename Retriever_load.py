import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import sqlite3
print("SQLite version used by modified Python:", sqlite3.sqlite_version)

from langchain_community.vectorstores import Chroma
import chromadb.utils.embedding_functions as embedding_functions
import ollama
from langchain_chroma import Chroma
import chromadb
from Embedding import OllamaEmbeddingsWrapper

embedding_wrapper = OllamaEmbeddingsWrapper()

def get_retriever():
    chroma_client = chromadb.PersistentClient(path="/workspaces/Self_Reflective_Rag/db")
    vectorstore_loaded = Chroma(
    client=chroma_client,
    collection_name="rag-chroma", 
    embedding_function=embedding_wrapper,
    )
    # Initialize the Chroma client
    #chroma_client = chromadb.PersistentClient(path="/workspaces/Self_Reflective_Rag/db")

    # Load the vector store from the specified directory
    #vectorstore_loaded = Chroma(
    #    client=chroma_client,
    #    collection_name="rag-chroma", 
    #    embedding_function=embedding_functions.OllamaEmbeddingFunction(
    #        url="http://localhost:11434/api/embeddings",
    #        model_name="nomic-embed-text",
    #    ),
    #    persist_directory="/workspaces/Self_Reflective_Rag/db"  # Specify your persist directory
    #)

    # Check if documents were loaded successfully
    print(f"Total documents in loaded vectorstore: {vectorstore_loaded._collection.count()}")
    return vectorstore_loaded.as_retriever()