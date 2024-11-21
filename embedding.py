import requests
from typing import List, Any
from llama_index.core.embeddings import BaseEmbedding

def ollama_embed(
    text:str, 
    url:str = "http://localhost:11434/api/embeddings",
    model:str = 'bge-m3') -> List[float]:
    data = {"model": model, "prompt": text}
    response = requests.post(url, json=data)
    return response.json()['embedding']

class BgeM3Embedding(BaseEmbedding):
    
    """
    Llama index compatible Embedding model(Use Ollama API call under the hood)
    
    TODO: query and text embeddings are the same, should be different (by GITHUB COPILOT)
    """

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "ollama_bge_m3"

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return ollama_embed(query)

    def _get_text_embedding(self, text: str) -> List[float]:
        return ollama_embed(text)

    def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
        return [ollama_embed(text) for text in texts]