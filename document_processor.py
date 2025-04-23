"""
document_processor.py: Handles document ingestion and text extraction
"""

import os
import tempfile
import PyPDF2
import pandas as pd
from docx import Document
from pptx import Presentation
import csv
import json
from typing import List, Dict, Any, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Class for processing various document types and extracting text content."""
    
    # Dans document_processor.py, modifiez le constructeur:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: The size of text chunks to create (réduit de 1000 à 500)
            chunk_overlap: The overlap between consecutive chunks (réduit de 200 à 100)
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Process a file and extract text content.
        
        Args:
            file_path: Path to the file to process
            
        Returns:
            List of dictionaries containing text chunks and metadata
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._process_pdf(file_path)
        elif file_extension == '.docx':
            return self._process_docx(file_path)
        elif file_extension == '.pptx':
            return self._process_pptx(file_path)
        elif file_extension == '.csv':
            return self._process_csv(file_path)
        elif file_extension == '.txt':
            return self._process_txt(file_path)
        elif file_extension == '.json':
            return self._process_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _process_pdf(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PDF files and extract text."""
        text_content = ""
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            num_pages = len(pdf_reader.pages)
            
            for page_num in range(num_pages):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text() + "\n\n"
        
        return self._create_chunks(text_content, {"source": file_path, "type": "pdf"})
    
    def _process_docx(self, file_path: str) -> List[Dict[str, Any]]:
        """Process DOCX files and extract text."""
        doc = Document(file_path)
        text_content = "\n\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        
        return self._create_chunks(text_content, {"source": file_path, "type": "docx"})
    
    def _process_pptx(self, file_path: str) -> List[Dict[str, Any]]:
        """Process PPTX files and extract text."""
        presentation = Presentation(file_path)
        text_content = ""
        
        for slide_num, slide in enumerate(presentation.slides):
            slide_text = f"Slide {slide_num+1}:\n"
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    slide_text += shape.text + "\n"
            text_content += slide_text + "\n\n"
        
        return self._create_chunks(text_content, {"source": file_path, "type": "pptx"})
    
    def _process_csv(self, file_path: str) -> List[Dict[str, Any]]:
        """Process CSV files and extract text."""
        df = pd.read_csv(file_path)
        # Convert to a more readable text format with column names
        text_content = df.to_string(index=False)
        
        return self._create_chunks(text_content, {"source": file_path, "type": "csv"})
    
    def _process_txt(self, file_path: str) -> List[Dict[str, Any]]:
        """Process TXT files and extract text."""
        with open(file_path, 'r', encoding='utf-8') as file:
            text_content = file.read()
        
        return self._create_chunks(text_content, {"source": file_path, "type": "txt"})
    
    def _process_json(self, file_path: str) -> List[Dict[str, Any]]:
        """Process JSON files and extract text."""
        with open(file_path, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
        
        # Convert JSON to string representation
        text_content = json.dumps(json_data, indent=2)
        
        return self._create_chunks(text_content, {"source": file_path, "type": "json"})
    
    def _create_chunks(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split text into chunks and add metadata.
        
        Args:
            text: The text to split
            metadata: Metadata about the source document
            
        Returns:
            List of dictionaries with text chunks and metadata
        """
        chunks = self.text_splitter.split_text(text)
        
        return [
            {
                "text": chunk,
                "metadata": {
                    **metadata,
                    "chunk_id": i
                }
            }
            for i, chunk in enumerate(chunks)
        ]
    
    def process_uploaded_file(self, uploaded_file) -> List[Dict[str, Any]]:
        """
        Process an uploaded file object (e.g., from Streamlit).
        
        Args:
            uploaded_file: File object from upload widget
            
        Returns:
            List of dictionaries with text chunks and metadata
        """
        # Create a temporary file to process
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        
        try:
            result = self.process_file(temp_file_path)
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
        return result

    def batch_process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported files in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            
        Returns:
            List of dictionaries with text chunks and metadata
        """
        all_chunks = []
        supported_extensions = ['.pdf', '.docx', '.pptx', '.csv', '.txt', '.json']
        
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_extension = os.path.splitext(file_path)[1].lower()
                
                if file_extension in supported_extensions:
                    try:
                        chunks = self.process_file(file_path)
                        all_chunks.extend(chunks)
                        print(f"Processed {file_path}: {len(chunks)} chunks created")
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")
        
        return all_chunks