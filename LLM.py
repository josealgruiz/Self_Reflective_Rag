#import sys
#__import__('pysqlite3')
#sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#import sqlite3
#print("SQLite version used by modified Python:", sqlite3.sqlite_version)

#from langchain_community.vectorstores import Chroma
#import chromadb.utils.embedding_functions as embedding_functions
#import ollama
#from langchain_chroma import Chroma
#import chromadb

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
#import requests
#from ollama_functions import OllamaFunctions
#from Embedding import OllamaEmbeddingsWrapper

#from langchain_ollama import ChatOllama

    
#embedding_wrapper = OllamaEmbeddingsWrapper()

#import the retriever
#print('Importing the retriever')
#chroma_client = chromadb.PersistentClient(path="/workspaces/Self_Reflective_Rag/db")
#vectorstore_loaded = Chroma(
#    client=chroma_client,
#    collection_name="rag-chroma", 
#    embedding_function=embedding_wrapper,
#    )

# Check if documents were loaded successfully
#print(f"Total documents in loaded vectorstore: {vectorstore_loaded._collection.count()}")

#retriever = vectorstore_loaded.as_retriever()

#print(hasattr(embedding_wrapper, 'embed_query'))  # Should print True
#print(hasattr(embedding_wrapper, 'embed_documents'))  # Should print True


# Data model
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )

# Prompt
system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ]
)

def retrieval_grader(llm,retriever):
    structured_llm_grader = llm.with_structured_output(GradeDocuments)

    retrieval_grader = grade_prompt | structured_llm_grader
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content
    retrieval_obtained = retrieval_grader.invoke({"question": question, "document": doc_txt})
    #print(retrieval_obtained)

    return retrieval_grader,docs,question,retrieval_obtained