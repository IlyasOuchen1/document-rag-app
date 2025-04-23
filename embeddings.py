"""
embeddings.py: Creates vector embeddings from text chunks
"""

import os
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EmbeddingsManager:
    """Class for generating embeddings from text."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: The name of the OpenAI embedding model to use
        """
        # Get API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            openai_api_key=api_key
        )
    
    def generate_embeddings(self, text_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            text_chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of dictionaries with text, metadata, and embeddings
        """
        # Extract text from chunks
        texts = [chunk["text"] for chunk in text_chunks]
        
        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)
        
        # Combine embeddings with original chunks
        embedded_chunks = []
        for i, chunk in enumerate(text_chunks):
            embedded_chunks.append({
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "embedding": embeddings[i]
            })
            
        return embedded_chunks
    
    def generate_query_embedding(self, query: str) -> List[float]:
        """
        Generate an embedding for a query string.
        
        Args:
            query: The query text
            
        Returns:
            Embedding vector
        """
        return self.embeddings.embed_query(query)