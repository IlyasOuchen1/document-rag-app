"""
embeddings.py: Creates vector embeddings from text chunks and images
"""

import os
from typing import List, Dict, Any, Union
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class EmbeddingsManager:
    """Class for generating embeddings from text and images."""
    
    def __init__(self, model_name: str = "text-embedding-3-small"):
        """
        Initialize the embeddings manager.
        
        Args:
            model_name: Name of the embedding model to use
        """
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Embeddings Manager initialized with model: {model_name}")
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64 string."""
        with Image.open(image_path) as img:
            buffered = BytesIO()
            img.save(buffered, format=img.format)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    def generate_image_embedding(self, image_path: str) -> List[float]:
        """
        Generate an embedding for an image using CLIP.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Image embedding vector
        """
        try:
            # Encode image
            base64_image = self._encode_image(image_path)
            
            # Get embedding from OpenAI
            response = self.client.embeddings.create(
                model="clip-vit-base-patch32",
                input=base64_image,
                encoding_format="float"
            )
            
            return response.data[0].embedding
        except Exception as e:
            print(f"Error generating image embedding: {e}")
            return None
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of text chunks.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of dictionaries with added embeddings
        """
        embedded_chunks = []
        
        for chunk in chunks:
            try:
                # Generate embedding for the text
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=chunk['text']
                )
                
                # Add embedding to the chunk
                embedded_chunk = chunk.copy()
                embedded_chunk['embedding'] = np.array(response.data[0].embedding)
                embedded_chunks.append(embedded_chunk)
                
            except Exception as e:
                logger.error(f"Error generating embedding: {str(e)}")
                continue
        
        logger.info(f"Generated embeddings for {len(embedded_chunks)} chunks")
        return embedded_chunks
    
    def generate_query_embedding(self, query: str) -> np.ndarray:
        """
        Generate embedding for a query string.
        
        Args:
            query: The query text
            
        Returns:
            Numpy array containing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=query
            )
            return np.array(response.data[0].embedding)
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise