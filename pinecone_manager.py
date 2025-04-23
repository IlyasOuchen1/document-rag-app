"""
pinecone_manager.py: Manages interactions with Pinecone vector database
"""

import os
import uuid
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class PineconeManager:
    """Class for managing interactions with Pinecone vector database."""
    
    def __init__(self, index_name: str = "document-rag"):
        """
        Initialize the Pinecone manager.
        
        Args:
            index_name: Name of the Pinecone index
        """
        # Get API key from environment variables
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY must be set")
            
        # Initialize Pinecone client
        self.pc = Pinecone(api_key=api_key)
        
        # Create index if it doesn't exist
        self.index_name = index_name
        self._create_index_if_not_exists()
        
        # Connect to the index
        self.index = self.pc.Index(self.index_name)
    
    def _create_index_if_not_exists(self, dimension: int = 1536):
        """
        Create Pinecone index if it doesn't already exist.
        
        Args:
            dimension: Dimension of the embedding vectors
        """
        # List existing indexes
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        
        # Check if index exists
        if self.index_name not in existing_indexes:
            print(f"Creating Pinecone index: {self.index_name}")
            # Create a serverless index
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws")  # Or "gcp" if preferred
            )
            print(f"Index {self.index_name} created")
    
    def upsert_vectors(self, embedded_chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert vectors to Pinecone index.
        
        Args:
            embedded_chunks: List of dictionaries with text, metadata, and embeddings
            batch_size: Size of batches for upserting
        """
        total_batches = len(embedded_chunks) // batch_size + (1 if len(embedded_chunks) % batch_size != 0 else 0)
        
        for i in tqdm(range(0, len(embedded_chunks), batch_size), total=total_batches, desc="Upserting to Pinecone"):
            batch = embedded_chunks[i:i+batch_size]
            
            # Prepare vectors for upserting
            vectors = []
            for chunk in batch:
                # Generate a unique ID for each vector
                vector_id = str(uuid.uuid4())
                
                vectors.append({
                    "id": vector_id,
                    "values": chunk["embedding"],
                    "metadata": {
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                })
            
            # Upsert batch to Pinecone
            self.index.upsert(vectors=vectors)
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in Pinecone.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Number of results to return
            filter: Optional filter for the search
            
        Returns:
            List of dictionaries with search results
        """
        # Query Pinecone index
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True,
            filter=filter
        )
        
        # Process and return results
        matches = []
        for match in results.matches:
            matches.append({
                "text": match.metadata["text"],
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"},
                "score": match.score
            })
            
        return matches
    
    def delete_vectors(self, filter: Dict[str, Any]):
        """
        Delete vectors from Pinecone index.
        
        Args:
            filter: Filter to select vectors to delete
        """
        # Delete vectors matching the filter
        self.index.delete(filter=filter)
    
    def clear_index(self):
        """Delete all vectors from the index."""
        self.index.delete(delete_all=True)
        print(f"Cleared all vectors from index {self.index_name}")