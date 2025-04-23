"""
local_vector_store.py: A local alternative to Pinecone using FAISS
"""

import os
import uuid
import faiss
import numpy as np
from typing import List, Dict, Any, Optional
import pickle

class LocalVectorStore:
    """A local vector store using FAISS as an alternative to Pinecone."""
    
    def __init__(self, index_name: str = "document-rag", dimension: int = 1536):
        """
        Initialize the local vector store.
        
        Args:
            index_name: Name of the index
            dimension: Dimension of the embedding vectors
        """
        self.index_name = index_name
        self.dimension = dimension
        self.vector_file = f"{index_name}_vectors.pkl"
        self.metadata_file = f"{index_name}_metadata.pkl"
        
        # Create or load index
        self._create_or_load_index()
        
        # Print index status for debugging
        print(f"LocalVectorStore initialized with {self.index.ntotal} vectors")
    
    def _create_or_load_index(self):
        """Create a new index or load an existing one."""
        if os.path.exists(self.vector_file) and os.path.exists(self.metadata_file):
            # Load existing index and metadata
            self.index = faiss.read_index(self.vector_file)
            with open(self.metadata_file, 'rb') as f:
                self.metadata = pickle.load(f)
            print(f"Loaded index with {self.index.ntotal} vectors")
        else:
            # Create new index and metadata
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = {}
            print(f"Created new index {self.index_name}")
    
    def _save_index(self):
        """Save the index and metadata to disk."""
        faiss.write_index(self.index, self.vector_file)
        with open(self.metadata_file, 'wb') as f:
            pickle.dump(self.metadata, f)
        print(f"Saved index with {self.index.ntotal} vectors")
    
    def upsert_vectors(self, embedded_chunks: List[Dict[str, Any]], batch_size: int = 100):
        """
        Upsert vectors to the index.
        
        Args:
            embedded_chunks: List of dictionaries with text, metadata, and embeddings
            batch_size: Size of batches for upserting (not used for local index)
        """
        if not embedded_chunks:
            print("Warning: No embedded chunks to upsert")
            return
            
        print(f"Upserting {len(embedded_chunks)} vectors to local store")
        
        for chunk in embedded_chunks:
            # Generate a unique ID for each vector
            vector_id = str(uuid.uuid4())
            
            # Convert embedding to numpy array and reshape for FAISS
            embedding = np.array(chunk["embedding"]).astype('float32').reshape(1, -1)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Store metadata with the same index
            idx = self.index.ntotal - 1
            self.metadata[idx] = {
                "text": chunk["text"],
                **chunk["metadata"]
            }
        
        # Save index and metadata
        self._save_index()
        print(f"Added {len(embedded_chunks)} vectors. Total vectors: {self.index.ntotal}")
    
    def similarity_search(
        self, 
        query_embedding: List[float], 
        top_k: int = None,  # Paramètre optionnel
        similarity_threshold: float = 0.85,  # Seuil de similarité augmenté
        filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform similarity search in the index.
        
        Args:
            query_embedding: Embedding vector of the query
            top_k: Optional limit on number of results (None = no limit)
            similarity_threshold: Minimum similarity score (0-1) for results
            filter: Optional filter for the search
            
        Returns:
            List of dictionaries with search results
        """
        # Check if index is empty
        if self.index.ntotal == 0:
            print("Warning: Vector store is empty - no documents have been indexed")
            return []
        
        print(f"Searching among {self.index.ntotal} vectors with threshold {similarity_threshold}")
        
        # Calculate max_k - how many results to retrieve initially
        max_k = self.index.ntotal if top_k is None else min(self.index.ntotal, max(top_k * 3, 50))
        
        # Convert query to numpy array
        query_np = np.array(query_embedding).astype('float32').reshape(1, -1)
        
        # Search index with larger k to filter by threshold later
        distances, indices = self.index.search(query_np, max_k)
        
        # Process results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # Valid index
                metadata = self.metadata.get(int(idx), {})
                
                # Calculate similarity score (convert distance to similarity)
                similarity = 1.0000 - float(distances[0][i]) / 100.0
                similarity = max(0.0000, min(1.0000, similarity))  # Clamp between 0 and 1
                
                # Skip results below threshold
                if similarity < similarity_threshold:
                    continue
                
                # Apply filter if provided
                if filter and not self._apply_filter(metadata, filter):
                    continue
                
                results.append({
                    "text": metadata.get("text", ""),
                    "metadata": {k: v for k, v in metadata.items() if k != "text"},
                    "score": similarity
                })
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        # Limit results if top_k is specified
        if top_k is not None:
            results = results[:top_k]
        
        print(f"Found {len(results)} matching results above threshold {similarity_threshold}")
        return results
    
    def _apply_filter(self, metadata: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        """
        Apply a simple filter to metadata.
        
        Args:
            metadata: Metadata to check
            filter: Filter to apply
            
        Returns:
            True if metadata matches filter, False otherwise
        """
        for key, value in filter.items():
            if key not in metadata or metadata[key] != value:
                return False
        return True
    
    def delete_vectors(self, filter: Dict[str, Any]):
        """
        Not fully implemented for local vector store.
        Would require rebuilding the index.
        """
        print("Warning: delete_vectors not fully implemented for local vector store")
    
    def clear_index(self):
        """Delete all vectors from the index."""
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = {}
        self._save_index()
        print(f"Cleared all vectors from index {self.index_name}")