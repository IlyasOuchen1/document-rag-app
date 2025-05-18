"""
app.py: Main application file for the Document RAG System
"""

import os
import streamlit as st
from typing import List, Dict, Any
import logging
from dotenv import load_dotenv
from document_processor import DocumentProcessor
from embeddings import EmbeddingsManager
from pinecone_manager import PineconeManager
from external_search import ExternalSearchManager
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
external_search_manager = ExternalSearchManager()

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
        return prompt_templates.get(lang, prompt_templates['en'])
    except:
        return prompt_templates['en']

def process_document(file_path: str) -> List[Dict[str, Any]]:
    """Process a document and return chunks with embeddings."""
    try:
        chunks = document_processor.process_file(file_path)
        logger.info(f"Processed document into {len(chunks)} chunks")
        
        if chunks:
            doc_language = detect(chunks[0]['text'])
            st.session_state.document_language = doc_language
            logger.info(f"Detected document language: {doc_language}")
        
        embedded_chunks = embeddings_manager.generate_embeddings(chunks)
        pinecone_manager.upsert_vectors(embedded_chunks)
        logger.info("Stored vectors in Pinecone")
        
        return embedded_chunks
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def query_document(query: str, search_mode: str = "internal", top_k: int = 5) -> List[Dict[str, Any]]:
    """Query the document using RAG."""
    try:
        context_parts = []
        sources = []

        # Recherche interne
        if search_mode in ["internal", "both"]:
            query_embedding = embeddings_manager.generate_query_embedding(query)
            internal_results = pinecone_manager.similarity_search(query_embedding, top_k=top_k)
            context_parts.extend([r["metadata"].get("text", "") for r in internal_results])
            sources.extend(internal_results)

        # Recherche externe
        if search_mode in ["external", "both"]:
            external_results = external_search_manager.search_all_sources(query, num_results=top_k)
            context_parts.extend([r["text"] for r in external_results])
            sources.extend(external_results)

        # Combiner les contextes
        context = "\n\n".join(context_parts)
        
        # Obtenir le template approprié
        template = get_prompt_template(context)
        prompt = ChatPromptTemplate.from_template(template)
        
        # Créer la chaîne RAG
        rag_chain = (
            {"context": lambda x: x["context"], "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        # Générer la réponse
        response = rag_chain.invoke({
            "context": context,
            "question": query
        })
        
        return {
            "response": response,
            "sources": sources
        }
    except Exception as e:
        logger.error(f"Error querying document: {str(e)}")
        raise

# Configure Streamlit page
st.set_page_config(
    page_title="Document RAG System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0;
        background-color: #1a1a1a;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
        background-color: #1a1a1a;
    }
    .upload-section {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #3d3d3d;
    }
    .query-section {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #3d3d3d;
    }
    .response-section {
        background-color: #2d2d2d;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        border: 1px solid #3d3d3d;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
        background-color: #3d3d3d;
        color: #ffffff;
        border: 1px solid #4d4d4d;
    }
    h1, h2, h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .stMarkdown {
        margin-bottom: 0;
        color: #ffffff;
    }
    .stSpinner {
        margin: 0;
    }
    .stSuccess, .stInfo, .stWarning {
        margin: 0;
        padding: 0.5rem;
        border-radius: 5px;
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: #ffffff;
    }
    .stExpander {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
    }
    .stExpander > div {
        background-color: #2d2d2d;
    }
    .stExpander > div > div {
        color: #ffffff;
    }
    .stFileUploader > div {
        background-color: #2d2d2d;
        border: 1px solid #3d3d3d;
    }
    .stFileUploader > div > div {
        color: #ffffff;
    }
    .stMarkdown p {
        color: #ffffff;
    }
    .stMarkdown a {
        color: #4CAF50;
    }
    .stMarkdown a:hover {
        color: #45a049;
    }
    .stMarkdown code {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    .stMarkdown pre {
        background-color: #3d3d3d;
        color: #ffffff;
    }
    .stMarkdown blockquote {
        border-left: 3px solid #4CAF50;
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/document.png", width=100)
    st.title("Document RAG System")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This system allows you to:
    - Upload documents (PDF, DOCX, TXT)
    - Ask questions about the content
    - Get AI-powered responses
    - View source references
    - Search in external sources
    """)
    st.markdown("---")
    st.markdown("### Supported Languages")
    st.markdown("""
    - English
    - French
    - Spanish
    - German
    """)

# Main content
st.markdown("<h1 style='text-align: center; margin-bottom: 2rem; color: #ffffff;'>Document RAG System</h1>", unsafe_allow_html=True)

# Initialize session state
if 'processed_documents' not in st.session_state:
    st.session_state.processed_documents = set()
if 'document_language' not in st.session_state:
    st.session_state.document_language = 'en'

# File upload section
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("### 📄 Upload Document")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx", "txt"])
if uploaded_file:
    file_path = os.path.join("uploads", uploaded_file.name)
    os.makedirs("uploads", exist_ok=True)
    
    if file_path not in st.session_state.processed_documents:
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        with st.spinner("Processing document..."):
            chunks = process_document(file_path)
            st.session_state.processed_documents.add(file_path)
            st.success(f"✅ Successfully processed {len(chunks)} chunks from document")
    else:
        st.info("📝 Document already processed. You can ask questions about it.")
st.markdown("</div>", unsafe_allow_html=True)

# Query section
st.markdown("<div class='query-section'>", unsafe_allow_html=True)
st.markdown("### 💭 Ask Questions")

# Search mode selection
search_mode = st.radio(
    "Select search mode:",
    ["internal", "external", "both"],
    format_func=lambda x: {
        "internal": "📚 Internal (Uploaded Documents)",
        "external": "🌐 External (Web & Wikipedia)",
        "both": "🔄 Both Sources"
    }[x],
    horizontal=True
)

query = st.text_input("Enter your question", key="query_input")

if query:
    if search_mode in ["internal", "both"] and not st.session_state.processed_documents:
        st.warning("⚠️ Please upload a document first for internal search.")
    else:
        with st.spinner("🔍 Searching for relevant information..."):
            result = query_document(query, search_mode=search_mode)
            
            st.markdown("<div class='response-section'>", unsafe_allow_html=True)
            st.markdown("### 📝 Response")
            st.write(result["response"])
            
            with st.expander("🔍 View Sources", expanded=False):
                for source in result["sources"]:
                    if "source" in source.get("metadata", {}):
                        # Source externe
                        source_text = source.get("text", "No text available")
                        source_type = source["metadata"]["source"]
                        source_url = source["metadata"].get("url", "")
                        st.markdown(f"""
                        <div style='background-color: #3d3d3d; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border: 1px solid #4d4d4d;'>
                            <p style='margin: 0; color: #ffffff;'>{source_text[:200]}...</p>
                            <p style='margin: 0.5rem 0 0 0; color: #888888;'>Source: {source_type}</p>
                            <p style='margin: 0.5rem 0 0 0; color: #888888;'><a href='{source_url}' target='_blank'>View Source</a></p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        # Source interne
                        source_text = source.get("metadata", {}).get("text", "No text available")
                        score = source.get("score", 0.0)
                        st.markdown(f"""
                        <div style='background-color: #3d3d3d; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; border: 1px solid #4d4d4d;'>
                            <p style='margin: 0; color: #ffffff;'>{source_text[:200]}...</p>
                            <p style='margin: 0.5rem 0 0 0; color: #888888;'>Relevance Score: {score:.2f}</p>
                        </div>
                        """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)