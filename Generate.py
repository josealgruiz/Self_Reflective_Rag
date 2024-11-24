### Generate

from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# Prompt
prompt = hub.pull("rlm/rag-prompt")

# LLM
llm = ChatOllama(model="llama3.2", temperature=0)
#llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm | StrOutputParser()

def generation(docs,question):
    # Run
    generation = rag_chain.invoke({"context": docs, "question": question})
    return(generation)
    