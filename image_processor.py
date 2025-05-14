import os
from PIL import Image
import pytesseract
import easyocr
import torch
from transformers import CLIPProcessor, CLIPModel
import numpy as np
from typing import List, Dict, Union, Tuple
import logging

class ImageProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from an image using OCR."""
        try:
            # First try with easyocr for better accuracy
            result = self.reader.readtext(image_path)
            text = ' '.join([item[1] for item in result])
            
            # If no text found, try pytesseract as fallback
            if not text.strip():
                text = pytesseract.image_to_string(Image.open(image_path))
            
            return text.strip()
        except Exception as e:
            logging.error(f"Error extracting text from image {image_path}: {str(e)}")
            return ""

    def get_image_embedding(self, image_path: str) -> np.ndarray:
        """Get CLIP embedding for an image."""
        try:
            image = Image.open(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            return image_features.numpy()[0]
        except Exception as e:
            logging.error(f"Error getting image embedding for {image_path}: {str(e)}")
            return np.zeros(512)  # Return zero vector as fallback

    def process_image(self, image_path: str) -> Dict[str, Union[str, np.ndarray]]:
        """Process an image and return both text and embedding."""
        return {
            "text": self.extract_text_from_image(image_path),
            "embedding": self.get_image_embedding(image_path)
        }

    def is_graph_or_chart(self, image_path: str) -> bool:
        """Detect if an image contains a graph or chart."""
        try:
            # Simple heuristic: check for presence of axes, grid lines, or common chart elements
            text = self.extract_text_from_image(image_path)
            keywords = ['axis', 'chart', 'graph', 'plot', 'x', 'y', 'value', 'data']
            return any(keyword in text.lower() for keyword in keywords)
        except Exception as e:
            logging.error(f"Error detecting graph in {image_path}: {str(e)}")
            return False

    def extract_graph_data(self, image_path: str) -> Dict:
        """Extract structured data from graphs and charts."""
        # This is a placeholder for more sophisticated graph analysis
        # In a real implementation, you might want to use specialized libraries
        # like OpenCV or scikit-image for graph detection and data extraction
        return {
            "is_graph": self.is_graph_or_chart(image_path),
            "extracted_text": self.extract_text_from_image(image_path),
            "embedding": self.get_image_embedding(image_path)
        } 