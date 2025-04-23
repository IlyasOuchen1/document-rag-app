"""
rag_engine.py: Combines retrieval and generation functionality
"""

import os
from typing import List, Dict, Any, Optional
from openai import OpenAI
from dotenv import load_dotenv
import logging
import tiktoken

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from embeddings import EmbeddingsManager
# Support both Pinecone and local vector store with the same interface
try:
    from pinecone_manager import PineconeManager
    VectorStore = PineconeManager
except ImportError:
    from local_vector_store import LocalVectorStore
    VectorStore = LocalVectorStore

# Load environment variables
load_dotenv()

class RAGEngine:
    """Class for combining retrieval and generation functionality."""
    
    def __init__(
        self,
        embeddings_manager: EmbeddingsManager,
        pinecone_manager: Any,  # Can be PineconeManager or LocalVectorStore
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 5000
    ):
        """
        Initialize the RAG engine.
        
        Args:
            embeddings_manager: Instance of EmbeddingsManager
            pinecone_manager: Instance of PineconeManager or LocalVectorStore
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
        self.vector_store = pinecone_manager  # Rename for clarity
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = 100000  # Limite de tokens pour le contexte
        
        logger.info(f"RAG Engine initialized with model: {model_name}")
    
    def _select_best_context(self, retrieved_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Sélectionne les meilleurs documents sans dépasser la limite de tokens.
        
        Args:
            retrieved_docs: Liste des documents récupérés
            
        Returns:
            Liste filtrée des meilleurs documents
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
        top_k: int = None,  # Paramètre optionnel
        similarity_threshold: float = 0.85,  # Seuil de similarité augmenté
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query using RAG.
        
        Args:
            query: The query text
            top_k: Optional limit on number of results (None = no limit)
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
        retrieved_docs = self.vector_store.similarity_search(
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
                "response": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question. Veuillez essayer d'uploader plus de documents ou reformuler votre question.",
                "sources": [],
                "contexts": []
            }
        
        # Filtrer les documents pour ne pas dépasser la limite de tokens
        filtered_docs = self._select_best_context(retrieved_docs)
        
        # Construct prompt with retrieved context
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(filtered_docs)])
        
        prompt = f"""
Vous êtes un assistant utile qui répond aux questions en vous basant sur les documents fournis.
Veuillez répondre à la question suivante en vous basant uniquement sur les informations contenues dans ces documents.
Si la réponse n'est pas dans les documents, dites-le - n'inventez pas d'information.

Documents:
{context}

Question: {query}

Réponse:
"""

        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "Vous êtes un assistant utile qui répond aux questions en vous basant sur le contexte fourni."},
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
            generated_text = f"J'ai rencontré une erreur lors de la génération d'une réponse. Veuillez réessayer. Erreur: {str(e)}"
        
        # Add source information
        sources = [
            {
                "source": doc["metadata"].get("source", "Source inconnue"),
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
        top_k: int = None,  # Paramètre optionnel
        similarity_threshold: float = 0.85,  # Seuil de similarité augmenté
        filter: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Chat interface for RAG.
        
        Args:
            messages: List of message dictionaries (role, content)
            top_k: Optional limit on number of results (None = no limit)
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
                "response": "Aucun message utilisateur trouvé.",
                "sources": [],
                "contexts": []
            }
        
        logger.info(f"Processing chat message: {latest_user_message}")
        
        # Generate embedding for the latest user message
        query_embedding = self.embeddings_manager.generate_query_embedding(latest_user_message)
        
        # Retrieve relevant documents
        retrieved_docs = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            filter=filter
        )
        
        logger.info(f"Retrieved {len(retrieved_docs)} documents for chat")
        
        # Check if we got any results
        if not retrieved_docs:
            logger.warning("No relevant documents found for chat")
            return {
                "response": "Je n'ai pas trouvé d'informations pertinentes pour répondre à votre question. Essayez d'uploader des documents d'abord.",
                "sources": [],
                "contexts": []
            }
        
        # Filtrer les documents pour ne pas dépasser la limite de tokens
        filtered_docs = self._select_best_context(retrieved_docs)
        
        # Construct system message with retrieved context
        context = "\n\n".join([f"Document {i+1}:\n{doc['text']}" for i, doc in enumerate(filtered_docs)])
        
        system_message = f"""
Vous êtes un assistant utile qui répond aux questions en vous basant sur les documents fournis.
Veuillez répondre en vous basant sur les informations contenues dans ces documents et l'historique de la conversation.
Si la réponse n'est pas dans les documents ou l'historique de la conversation, dites-le - n'inventez pas d'information.

Contextes de documents pertinents:
{context}
"""
        
        # Prepare messages for the API
        chat_messages = [
            {"role": "system", "content": system_message},
            *messages  # Include conversation history
        ]
        
        # Generate response
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat_messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract and return response
            generated_text = response.choices[0].message.content
            logger.info("Generated chat response successfully")
            
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            generated_text = f"J'ai rencontré une erreur lors de la génération d'une réponse. Veuillez réessayer. Erreur: {str(e)}"
        
        # Add source information
        sources = [
            {
                "source": doc["metadata"].get("source", "Source inconnue"),
                "score": doc.get("score", 0)
            }
            for doc in filtered_docs
        ]
        
        return {
            "response": generated_text,
            "sources": sources,
            "contexts": [doc["text"] for doc in filtered_docs]
        }