"""
app.py: Main application file for the Document RAG system
"""

import os
import streamlit as st
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from embeddings import EmbeddingsManager
from pinecone_manager import PineconeManager
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langdetect import detect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize components
document_processor = DocumentProcessor()
embeddings_manager = EmbeddingsManager()
pinecone_manager = PineconeManager()

# Initialize LangChain components
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    streaming=True
)

# Define prompt templates for different languages
prompt_templates = {
    'en': """You are a helpful AI assistant. Use the following context to answer the question:

Context: {context}

Question: {question}

Answer: Let me help you with that.""",
    
    'fr': """Vous êtes un assistant IA utile. Utilisez le contexte suivant pour répondre à la question :

Contexte : {context}

Question : {question}

Réponse : Je vais vous aider avec cela.""",
    
    'es': """Eres un asistente de IA útil. Usa el siguiente contexto para responder a la pregunta:

Contexto: {context}

Pregunta: {question}

Respuesta: Déjame ayudarte con eso.""",
    
    'de': """Sie sind ein hilfreicher KI-Assistent. Verwenden Sie den folgenden Kontext, um die Frage zu beantworten:

Kontext: {context}

Frage: {question}

Antwort: Lassen Sie mich Ihnen dabei helfen."""
}

def get_prompt_template(text: str) -> str:
    """Detect language and return appropriate prompt template."""
    try:
        lang = detect(text)
        return prompt_templates.get(lang, prompt_templates['en'])  # Default to English if language not supported
    except:
        return prompt_templates['en']  # Default to English if detection fails

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """Process a document and return chunks with embeddings."""
    try:
        # Process document
        chunks = document_processor.process_file(file_path)
        logger.info(f"Processed document into {len(chunks)} chunks")
        
        # Detect language from first chunk
        if chunks:
            doc_language = detect(chunks[0]['text'])
            st.session_state.document_language = doc_language
            logger.info(f"Detected document language: {doc_language}")
        
        # Generate embeddings
        embedded_chunks = embeddings_manager.generate_embeddings(chunks)
        
        # Store in Pinecone
        pinecone_manager.upsert_vectors(embedded_chunks)
        logger.info("Stored vectors in Pinecone")
        
        return embedded_chunks
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def query_document(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Query the document using RAG."""
    try:
        # Generate query embedding
        query_embedding = embeddings_manager.generate_query_embedding(query)
        
        # Search in Pinecone
        results = pinecone_manager.similarity_search(query_embedding, top_k=top_k)
        
        # Format context
        context = "\n\n".join([r["metadata"].get("text", "") for r in results])
        
        # Get appropriate prompt template based on document language
        template = get_prompt_template(context)
        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the RAG chain with the appropriate prompt
        rag_chain = (
            {"context": lambda x: x["context"], "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Generate response using RAG chain
        response = rag_chain.invoke({
            "context": context,
            "question": query
        })
        
        return {
            "response": response,
            "sources": results
        }
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise

# Streamlit UI
st.title("Document RAG System")

# Initialize session state for processed documents and language
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = set()
if 'document_language' not in st.session_state:
    st.session_state.document_language = 'en'

# File upload
uploaded_file = st.file_uploader("Upload a document", type=["pdf", "docx", "txt"])
if uploaded_file:
    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    
    # Check if document was already processed
    if file_path not in st.session_state.processed_documents:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Process document
        with st.spinner("Processing document..."):
            chunks = process_document(file_path)
            st.session_state.processed_documents.add(file_path)
            st.success(f"Processed {len(chunks)} chunks from document")
    else:
        st.info("Document already processed. You can ask questions about it.")

# Query interface
query = st.text_input("Ask a question about the document")
if query:
    if not st.session_state.processed_documents:
        st.warning("Please upload a document first.")
    else:
        with st.spinner("Searching..."):
            result = query_document(query)
            st.write("Answer:", result["response"])
            
            # Show sources in an expandable section
            with st.expander("Show Sources", expanded=False):
                for source in result["sources"]:
                    # Safely get text from metadata with a default value
                    source_text = source.get('metadata', {}).get('text', 'No text available')
                    score = source.get('score', 0.0)
                    st.write(f"- {source_text[:200]}... (Score: {score:.2f})")