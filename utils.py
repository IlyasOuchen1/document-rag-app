"""
utils.py: Helper functions and utilities
"""

import os
import json
from typing import List, Dict, Any
from datetime import datetime

def save_processed_data(data: List[Dict[str, Any]], output_dir: str, file_name: str = None):
    """
    Save processed data to a JSON file.
    
    Args:
        data: Data to save
        output_dir: Directory to save the file in
        file_name: Name of the file (without extension)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate file name if not provided
    if not file_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"processed_data_{timestamp}"
    
    # Ensure file has .json extension
    if not file_name.endswith(".json"):
        file_name += ".json"
    
    # Save data
    file_path = os.path.join(output_dir, file_name)
    
    # We need to remove embeddings as they are not JSON serializable
    serializable_data = []
    for item in data:
        # Create a copy without the embedding
        serializable_item = {
            "text": item["text"],
            "metadata": item["metadata"]
        }
        serializable_data.append(serializable_item)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"Data saved to {file_path}")
    return file_path

def load_processed_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Load processed data from a JSON file.
    
    Args:
        file_path: Path to the JSON file
    
    Returns:
        List of dictionaries with text and metadata
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def get_file_extension(file_path: str) -> str:
    """
    Get the file extension from a file path.
    
    Args:
        file_path: Path to the file
    
    Returns:
        File extension (lowercase, with dot)
    """
    return os.path.splitext(file_path)[1].lower()

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """
    Format source information into a readable string.
    
    Args:
        sources: List of dictionaries with source information
        
    Returns:
        Formatted string
    """
    if not sources:
        return "No sources found."
    
    # Sort sources by relevance score
    sorted_sources = sorted(sources, key=lambda x: x.get("score", 0), reverse=True)
    
    formatted_text = "Sources:\n"
    for i, source in enumerate(sorted_sources):
        file_name = os.path.basename(source["source"])
        score = source.get("score", 0)
        formatted_text += f"{i+1}. {file_name} (Relevance: {score:.2f})\n"
    
    return formatted_text

def format_chunks(contexts):
    """
    Format the retrieved text chunks into a readable string.
    
    Args:
        contexts: List of text chunks used for generating the response
        
    Returns:
        Formatted string
    """
    if not contexts:
        return "No text chunks were used."
    
    formatted_text = "Text chunks used for this response:\n\n"
    for i, chunk in enumerate(contexts):
        formatted_text += f"**Chunk {i+1}:**\n"
        # Limiter la longueur du chunk affiché pour éviter des affichages trop longs
        if len(chunk) > 500:
            formatted_text += f"{chunk[:500]}...\n\n"
        else:
            formatted_text += f"{chunk}\n\n"
    
    return formatted_text