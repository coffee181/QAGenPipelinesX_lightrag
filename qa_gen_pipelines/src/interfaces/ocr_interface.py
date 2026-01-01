"""OCR interface definition."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
from ..models.document import Document


class OCRInterface(ABC):
    """Abstract interface for OCR implementations."""
    
    @abstractmethod
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            OCRError: If extraction fails
        """
        pass
    
    @abstractmethod
    def process_pdf_to_document(self, pdf_path: Path, output_dir: Optional[Path] = None) -> Document:
        """
        Process PDF and create Document object.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document object with extracted content and metadata
            
        Raises:
            OCRError: If processing fails
        """
        pass
    
    @abstractmethod
    def process_batch(self, pdf_paths: List[Path], output_dir: Path) -> List[Document]:
        """
        Process multiple PDF files in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            output_dir: Directory to save extracted text files
            
        Returns:
            List of processed Document objects
            
        Raises:
            OCRError: If batch processing fails
        """
        pass
    
    @abstractmethod
    def is_supported_format(self, file_path: Path) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        pass


class OCRError(Exception):
    """Custom exception for OCR errors."""
    pass 