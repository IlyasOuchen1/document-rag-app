import streamlit as st
from dotenv import load_dotenv
import os

# Try to import components
try:
    from document_processor import DocumentProcessor
    document_processor_loaded = True
except Exception as e:
    document_processor_loaded = False
    document_processor_error = str(e)

try:
    from embeddings import EmbeddingsManager
    embeddings_manager_loaded = True
except Exception as e:
    embeddings_manager_loaded = False
    embeddings_manager_error = str(e)

try:
    from pinecone_manager import PineconeManager
    pinecone_manager_loaded = True
except Exception as e:
    pinecone_manager_loaded = False
    pinecone_manager_error = str(e)

try:
    from rag_engine import RAGEngine
    rag_engine_loaded = True
except Exception as e:
    rag_engine_loaded = False
    rag_engine_error = str(e)

# Load environment variables
load_dotenv()

# Basic page setup
st.title("Document RAG App")
st.write("This is a test app")

# Display component loading status
st.write(f"Document Processor loaded: {'✅' if document_processor_loaded else '❌'}")
if not document_processor_loaded:
    st.error(f"Error loading Document Processor: {document_processor_error}")

st.write(f"Embeddings Manager loaded: {'✅' if embeddings_manager_loaded else '❌'}")
if not embeddings_manager_loaded:
    st.error(f"Error loading Embeddings Manager: {embeddings_manager_error}")

st.write(f"Pinecone Manager loaded: {'✅' if pinecone_manager_loaded else '❌'}")
if not pinecone_manager_loaded:
    st.error(f"Error loading Pinecone Manager: {pinecone_manager_error}")

st.write(f"RAG Engine loaded: {'✅' if rag_engine_loaded else '❌'}")
if not rag_engine_loaded:
    st.error(f"Error loading RAG Engine: {rag_engine_error}")

# Create tabs
tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])

with tab1:
    st.write("This is tab 1")
    user_input = st.text_input("Enter some text")
    if user_input:
        st.write(f"You entered: {user_input}")

with tab2:
    st.write("This is tab 2")
    file_upload = st.file_uploader("Upload a file", type=["txt", "pdf"])
    if file_upload:
        st.write(f"You uploaded: {file_upload.name}")

# Display environment info (without showing actual keys)
openai_key = os.getenv("OPENAI_API_KEY")
pinecone_key = os.getenv("PINECONE_API_KEY")

st.write(f"OpenAI API Key set: {'Yes' if openai_key else 'No'}")
st.write(f"Pinecone API Key set: {'Yes' if pinecone_key else 'No'}")