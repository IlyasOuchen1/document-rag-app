# check_vectors.py
from local_vector_store import LocalVectorStore
import os

def check_vector_store():
    """Check if the vector store contains any data."""
    
    # Try to load the vector store
    try:
        vector_store = LocalVectorStore()
        
        # Check if vector files exist
        vector_file = "document-rag_vectors.pkl"
        metadata_file = "document-rag_metadata.pkl"
        
        print(f"Vector file exists: {os.path.exists(vector_file)}")
        print(f"Metadata file exists: {os.path.exists(metadata_file)}")
        
        # Check vector count
        print(f"Total vectors in store: {vector_store.index.ntotal}")
        
        # Print sample metadata if available
        if vector_store.index.ntotal > 0:
            print("\nSample metadata entries:")
            sample_count = min(3, len(vector_store.metadata))
            for i, (idx, metadata) in enumerate(list(vector_store.metadata.items())[:sample_count]):
                print(f"Item {i+1}:")
                print(f"  Source: {metadata.get('source', 'Unknown')}")
                print(f"  Text snippet: {metadata.get('text', 'No text')[:100]}...")
    except Exception as e:
        print(f"Error checking vector store: {e}")

if __name__ == "__main__":
    check_vector_store()