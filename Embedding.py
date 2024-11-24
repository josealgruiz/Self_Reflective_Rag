import ollama

class OllamaEmbeddingsWrapper:
    def __init__(self, model='nomic-embed-text'):
        self.model = model

    def embed_documents(self, texts):
        return [ollama.embeddings(model=self.model, prompt=text)['embedding'] for text in texts]

    def embed_query(self, text):
        return ollama.embeddings(model=self.model, prompt=text)['embedding']
