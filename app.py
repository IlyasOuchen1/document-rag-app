# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import tempfile

from document_processor import DocumentProcessor
from embeddings import EmbeddingsManager
from pinecone_manager import PineconeManager
from rag_engine import RAGEngine
from utils import format_sources, format_chunks

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document RAG App",
    page_icon="📚",
    layout="wide"
)

# Initialize components
@st.cache_resource
def initialize_components():
    embeddings_manager = EmbeddingsManager()
    from local_vector_store import LocalVectorStore
    vector_store = LocalVectorStore()
    rag_engine = RAGEngine(
        embeddings_manager=embeddings_manager,
        pinecone_manager=vector_store  # We're still using the same parameter name
    )
    document_processor = DocumentProcessor()
    
    return embeddings_manager, vector_store, rag_engine, document_processor

embeddings_manager, pinecone_manager, rag_engine, document_processor = initialize_components()

# Set up the Streamlit app
st.title("📚 Document RAG Application")
st.markdown("Query multiple document types using AI with Retrieval Augmented Generation")

# Create tabs
tab1, tab2 = st.tabs(["💬 Chat with Documents", "📤 Upload Documents"])

# Tab 1: Chat with Documents
with tab1:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("Sources"):
                    st.markdown(message["sources"])
            if "chunks" in message:
                with st.expander("Text Chunks Used"):
                    st.markdown(message["chunks"])
    
    # Input for new query
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Format messages for the RAG engine
                    messages = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages
                    ]
                    
                    # Process query
                    response = rag_engine.chat(messages)
                    
                    # Check if response is None
                    if response is None:
                        st.error("No response received. The system may still be initializing or no relevant documents were found.")
                        sources_text = "No sources available."
                        chunks_text = "No text chunks available."
                        response_text = "I couldn't find relevant information to answer your question. Try uploading some documents first."
                    else:
                        # Display response
                        response_text = response.get("response", "No response content available.")
                        st.markdown(response_text)
                        
                        # Display sources
                        sources_text = "No sources found."
                        if response.get("sources", []):
                            sources_text = format_sources(response["sources"])
                            with st.expander("Sources"):
                                st.markdown(sources_text)
                        
                        # Display text chunks
                        chunks_text = "No text chunks available."
                        if response.get("contexts", []):
                            chunks_text = format_chunks(response.get("contexts", []))
                            with st.expander("Text Chunks Used"):
                                st.markdown(chunks_text)
                
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    sources_text = "No sources available due to error."
                    chunks_text = "No text chunks available due to error."
                    response_text = f"I encountered an error while processing your request: {str(e)}"
                    st.markdown(response_text)
                
                # Add assistant response to chat history
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": response_text,
                    "sources": sources_text,
                    "chunks": chunks_text
                })

# Tab 2: Upload Documents
with tab2:
    st.header("Upload Documents")
    st.write("Upload documents to be processed and added to the knowledge base.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, PPTX, CSV, TXT, JSON)", 
            accept_multiple_files=True,
            type=["pdf", "docx", "pptx", "csv", "txt", "json"]
        )
    
    with col2:
        st.info("""
        **Supported File Types:**
        - PDF (.pdf)
        - Word Documents (.docx)
        - PowerPoint (.pptx)
        - CSV (.csv)
        - Text Files (.txt)
        - JSON (.json)
        """)
    
    if uploaded_files:
        if st.button("Process Documents"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            total_files = len(uploaded_files)
            for i, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {uploaded_file.name}...")
                
                # Process file
                text_chunks = document_processor.process_uploaded_file(uploaded_file)
                
                # Generate embeddings
                embedded_chunks = embeddings_manager.generate_embeddings(text_chunks)
                
                # Store in Pinecone
                pinecone_manager.upsert_vectors(embedded_chunks)
                
                # Update progress
                progress_bar.progress((i + 1) / total_files)
                status_text.text(f"Processed {uploaded_file.name}: {len(text_chunks)} chunks created")
            
            status_text.text("All documents processed successfully!")
            st.success(f"✅ Successfully processed {total_files} documents!")
    
    # Directory processing (for local development)
    st.header("Process Local Directory")
    st.write("For local development: Process a directory of documents.")
    
    dir_path = st.text_input("Directory path")
    
    if dir_path and st.button("Process Directory"):
        if os.path.isdir(dir_path):
            with st.spinner("Processing directory..."):
                # Process documents
                text_chunks = document_processor.batch_process_directory(dir_path)
                st.write(f"Processed {len(text_chunks)} text chunks")
                
                # Generate embeddings
                embedded_chunks = embeddings_manager.generate_embeddings(text_chunks)
                st.write(f"Generated embeddings for {len(embedded_chunks)} chunks")
                
                # Store in Pinecone
                pinecone_manager.upsert_vectors(embedded_chunks)
                st.success("Directory processed successfully!")
        else:
            st.error("Directory not found")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using LangChain, OpenAI, and Pinecone")