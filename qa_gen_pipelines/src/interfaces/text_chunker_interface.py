"""Text chunker interface definition."""

from abc import ABC, abstractmethod
from typing import List
from ..models.document import Document, DocumentChunk


class TextChunkerInterface(ABC):
    """Abstract interface for text chunking implementations."""
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Split text into chunks.
        
        Args:
            text: Text content to be chunked
            document_id: ID of the source document
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            ChunkingError: If chunking fails
        """
        pass
    
    @abstractmethod
    def chunk_document(self, document: Document) -> List[DocumentChunk]:
        """
        Split document content into chunks.
        
        Args:
            document: Document object to be chunked
            
        Returns:
            List of DocumentChunk objects
            
        Raises:
            ChunkingError: If chunking fails
        """
        pass
    
    @abstractmethod
    def get_optimal_chunk_size(self, text: str) -> int:
        """
        Calculate optimal chunk size for given text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Recommended chunk size
        """
        pass
    
    @abstractmethod
    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate if chunks are properly formed.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            True if all chunks are valid, False otherwise
        """
        pass


class ChunkingError(Exception):
    """Custom exception for text chunking errors."""
    pass 