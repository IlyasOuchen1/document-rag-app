"""
rag_engine.py: Combines retrieval and generation functionality
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
import tiktoken
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from embeddings import EmbeddingsManager
from pinecone_manager import PineconeManager

# Load environment variables
load_dotenv()

class RAGEngine:
    """Class for combining retrieval and generation functionality."""
    
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        pinecone_manager: PineconeManager,
        model_name: str = "gpt-4",
        temperature: float = 0.3,
        max_tokens: int = 5000
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embeddings_manager: Instance of EmbeddingsManager
            pinecone_manager: Instance of PineconeManager
            model_name: Name of the generation model to use
            temperature: Temperature parameter for generation
            max_tokens: Maximum tokens for generation
        """
        # Get OpenAI API key from environment variables
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
            
        self.client = OpenAI(api_key=api_key)
        self.embeddings_manager = embeddings_manager
        self.pinecone_manager = pinecone_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = 100000  # Context token limit
        
        logger.info(f"RAG Engine initialized with model: {model_name}")
    
    def _select_best_context(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Selects the best documents without exceeding token limit.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Filtered list of best documents
        """
        try:
            enc = tiktoken.encoding_for_model(self.model_name)
        except:
            enc = tiktoken.get_encoding("cl100k_base")  # Fallback encoding
        
        sorted_docs = sorted(retrieved_docs, key=lambda x: x.get("score", 0), reverse=True)
        
        selected_docs = []
        total_tokens = 0
        
        for doc in sorted_docs:
            doc_tokens = len(enc.encode(doc["text"]))
                
            if total_tokens + doc_tokens > self.max_context_tokens:
                break
            
            selected_docs.append(doc)
            total_tokens += doc_tokens
        
        logger.info(f"Selected {len(selected_docs)}/{len(retrieved_docs)} documents totaling {total_tokens} tokens")
        return selected_docs
    
    def process_query(
        self, 
        query: str, 
        top_k: int = None,
        similarity_threshold: float = 0.85,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query using RAG.
        
        Args:
            query: The query text
            top_k: Optional limit on number of results
            similarity_threshold: Minimum similarity score for results
            filter: Optional filter for retrieval
            
        Returns:
            Dictionary with generated response and retrieved contexts
        """
        logger.info(f"Processing query: {query}")
        
        # Generate embedding for query
        query_embedding = self.embeddings_manager.generate_query_embedding(query)
        logger.info("Generated query embedding")
        
        # Retrieve relevant documents
        retrieved_docs = self.pinecone_manager.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents")
        
        # Check if we got any results
        if not retrieved_docs:
            logger.warning("No relevant documents found")
            return {
                "query": query,
                "response": "I couldn't find any relevant information to answer your question. Please try uploading more documents or rephrasing your question.",
                "sources": [],
                "contexts": []
            }
        
        # Filter documents to not exceed token limit
        filtered_docs = self._select_best_context(retrieved_docs)
        
        # Construct prompt with retrieved context
        context_parts = []
        for i, doc in enumerate(filtered_docs):
            context_parts.append(f"Document {i+1}:\n{doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""
You are a helpful assistant that answers questions based on the provided documents.
Please answer the following question based only on the information contained in these documents.
If the answer is not in the documents, say so - do not make up information.

Documents:
{context}

Question: {query}

Answer:
"""

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return response
            generated_text = response.choices[0].message.content
            logger.info("Generated response successfully")
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            generated_text = f"I encountered an error while generating a response. Please try again. Error: {str(e)}"
        
        # Add source information
        sources = [
            {
                "source": doc["metadata"].get("source", "Unknown source"),
                "type": doc["metadata"].get("type", "text"),
                "score": doc.get("score", 0)
            }
            for doc in filtered_docs
        ]
        
        return {
            "query": query,
            "response": generated_text,
            "sources": sources,
            "contexts": [doc["text"] for doc in filtered_docs]
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        top_k: int = None,
        similarity_threshold: float = 0.85,
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chat interface for RAG.
        
        Args:
            messages: List of message dictionaries (role, content)
            top_k: Optional limit on number of results
            similarity_threshold: Minimum similarity score for results
            filter: Optional filter for retrieval
            
        Returns:
            Dictionary with generated response and retrieved contexts
        """
        # Extract user's latest message
        latest_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            None
        )
        
        if not latest_user_message:
            logger.warning("No user message found")
            return {
                "response": "No user message found.",
                "sources": [],
                "contexts": []
            }
        
        # Process the query
        return self.process_query(
            query=latest_user_message,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter
        )