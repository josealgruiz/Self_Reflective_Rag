LANGCHAIN_TRACING_V2= True
LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
LANGCHAIN_API_KEY="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"
LANGCHAIN_PROJECT="self-rag-jose"

import LLM as Llm
import Generate
import Hallucination_grader
import Answer_grader
import Question_rewriter

from langchain_ollama import ChatOllama


#define the llm model to be used
llm = ChatOllama(model="llama3.2", temperature=0)

docs,question,retrieval_g = Llm.retrieval_grader()
print(f'docs: {docs},question: {question}, retrival grader: {retrieval_g}')

generation = Generate.generation(docs,question)
print(f'generation: {generation}')

halu = Hallucination_grader.hallu_grader(llm,docs,generation)
print(f'hallucination grader: {halu}')

ans = Answer_grader.answer_grader(llm,question,generation)
print(f'Answer grader: {ans}')

new_question = Question_rewriter.question_rewriter(llm,question)
print(f'new question: {new_question}')
