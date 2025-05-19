# Document RAG System

A powerful Document Retrieval-Augmented Generation (RAG) system that allows users to upload documents and ask questions about their content. The system uses advanced language models and vector embeddings to provide accurate, context-aware responses in the same language as the source document.

## Features

- 📄 Support for multiple document formats (PDF, DOCX, TXT)
- 🌍 Automatic language detection and response in the same language
- 🔍 Semantic search using vector embeddings
- 💾 Efficient document chunking and storage
- 🔄 Persistent storage using Pinecone vector database
- 🎯 Accurate answers with source references
- 🎨 Clean and intuitive Streamlit interface

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:
```bash
git clone https://github.com/IlyasOuchen1/document-rag-app
cd document-rag-app
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

## Project Structure

```
document-rag-app/
├── app.py                 # Main application file
├── document_processor.py  # Document processing and chunking
├── embeddings.py         # Embeddings generation
├── pinecone_manager.py   # Pinecone database management
├── requirements.txt      # Project dependencies
└── uploads/             # Directory for uploaded documents
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a document using the file uploader

4. Ask questions about the document in the text input field

5. View the AI's response and optionally check the sources by clicking "Show Sources"

## How It Works

1. **Document Processing**:
   - Documents are uploaded and processed into manageable chunks
   - Text is extracted and prepared for embedding generation

2. **Embedding Generation**:
   - Text chunks are converted into vector embeddings using OpenAI's embedding model
   - Embeddings capture semantic meaning for better search results

3. **Vector Storage**:
   - Embeddings are stored in Pinecone vector database
   - Efficient similarity search is enabled

4. **Query Processing**:
   - User questions are converted to embeddings
   - Similar chunks are retrieved from the database
   - Context is provided to the language model for accurate responses

5. **Response Generation**:
   - Language model generates responses based on retrieved context
   - Responses are provided in the same language as the source document

## Dependencies

- streamlit: Web interface
- langchain: RAG implementation
- openai: Language model and embeddings
- pinecone-client: Vector database
- python-dotenv: Environment variable management
- PyPDF2: PDF processing
- python-docx: DOCX processing
- langdetect: Language detection

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the language models and embeddings
- Pinecone for the vector database service
- Streamlit for the web interface framework 
