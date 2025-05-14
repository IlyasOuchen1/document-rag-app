"""
pinecone_manager.py: Manages interactions with Pinecone vector database
"""

import os
from typing import List, Dict, Any
import numpy as np
from dotenv import load_dotenv
import logging
from pinecone import Pinecone  # Updated import
from tqdm import tqdm
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class PineconeManager:
    """Class for managing interactions with Pinecone vector database."""
    
    def __init__(self, index_name: str = "agent-new"):
        """Initialize Pinecone manager with API key and index name."""
        self.api_key = os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.index_name = index_name
        # Create Pinecone instance instead of using pinecone.init
        self.pc = Pinecone(api_key=self.api_key)
        self._init_index()
        logger.info(f"PineconeManager initialized with index: {index_name}")
    
    def _init_index(self):
        """Initialize or get existing Pinecone index."""
        try:
            # List all indexes
            indexes = self.pc.list_indexes()
            index_names = [index.name for index in indexes]
            
            if self.index_name not in index_names:
                # Create new index - correctly structured for Pinecone 6.0.2
                self.pc.create_index(
                    name=self.index_name,
                    spec={
                        "dimension": 1536,  # OpenAI embedding dimension
                        "metric": "cosine",
                        "serverless": {
                            "cloud": "aws",
                            "region": "us-west-2"  # Choose your preferred region
                        }
                    }
                )
                logger.info(f"Created new Pinecone index: {self.index_name}")
            else:
                logger.info(f"Using existing Pinecone index: {self.index_name}")
            
            # Get index object through Pinecone instance
            self.index = self.pc.Index(self.index_name)
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise
    
    def upsert_vectors(self, vectors: List[Dict[str, Any]]):
        """Upsert vectors to Pinecone index."""
        try:
            # Prepare vectors for upsert
            upsert_data = []
            for i, vec in enumerate(vectors):
                if not isinstance(vec["embedding"], np.ndarray):
                    vec["embedding"] = np.array(vec["embedding"])
                
                # Create metadata with text included
                metadata = vec.get("metadata", {}).copy()
                metadata["text"] = vec.get("text", "")  # Add text to metadata
                
                upsert_data.append({
                    "id": str(uuid.uuid4()),  # Using UUID to ensure uniqueness
                    "values": vec["embedding"].tolist(),
                    "metadata": metadata
                })
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(upsert_data), batch_size):
                batch = upsert_data[i:i + batch_size]
                self.index.upsert(vectors=batch)
            
            logger.info(f"Successfully upserted {len(vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error upserting vectors to Pinecone: {str(e)}")
            raise
    
    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, 
                         filter: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar vectors in Pinecone index."""
        try:
            if not isinstance(query_embedding, np.ndarray):
                query_embedding = np.array(query_embedding)
            
            # Updated query method call with flattened parameters
            results = self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                filter=filter
            )
            
            # Format results
            formatted_results = []
            for match in results.matches:
                formatted_results.append({
                    "metadata": match.metadata,
                    "score": match.score
                })
            
            logger.info(f"Found {len(formatted_results)} similar vectors")
            return formatted_results
        except Exception as e:
            logger.error(f"Error performing similarity search in Pinecone: {str(e)}")
            raise
    
    def delete_vectors(self, filter: Dict[str, Any]):
        """
        Delete vectors from Pinecone index.
        
        Args:
            filter: Filter to select vectors to delete
        """
        # Delete vectors matching the filter
        self.index.delete(filter=filter)
    
    def clear_index(self):
        """Clear all vectors from the index."""
        try:
            self.index.delete(delete_all=True)
            logger.info("Cleared all vectors from Pinecone index")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {str(e)}")
            raise