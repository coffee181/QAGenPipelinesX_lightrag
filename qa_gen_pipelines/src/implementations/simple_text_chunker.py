"""Simple text chunker implementation."""

import re
from typing import List
from loguru import logger

from ..interfaces.text_chunker_interface import TextChunkerInterface, ChunkingError
from ..models.document import Document, DocumentChunk
from ..utils.config import ConfigManager


class SimpleTextChunker(TextChunkerInterface):
    """Simple text chunking implementation."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize text chunker.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.max_chunk_size = config.get("text_chunker.max_chunk_size", 2000)
        self.overlap_size = config.get("text_chunker.overlap_size", 200)
        self.chunk_on_sentences = config.get("text_chunker.chunk_on_sentences", True)
        
        logger.info(f"Text chunker initialized: max_size={self.max_chunk_size}, "
                   f"overlap={self.overlap_size}, sentences={self.chunk_on_sentences}")
    
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
        try:
            if not text.strip():
                return []
            
            logger.info(f"Chunking text for document: {document_id}")
            
            if self.chunk_on_sentences:
                chunks = self._chunk_by_sentences(text, document_id)
            else:
                chunks = self._chunk_by_characters(text, document_id)
            
            logger.info(f"Created {len(chunks)} chunks for document: {document_id}")
            return chunks
            
        except Exception as e:
            raise ChunkingError(f"Failed to chunk text for {document_id}: {e}")
    
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
        return self.chunk_text(document.content, str(document.file_path))
    
    def get_optimal_chunk_size(self, text: str) -> int:
        """
        Calculate optimal chunk size for given text.
        
        Args:
            text: Text content to analyze
            
        Returns:
            Recommended chunk size
        """
        text_length = len(text)
        
        # If text is shorter than max chunk size, return text length
        if text_length <= self.max_chunk_size:
            return text_length
        
        # Calculate number of chunks needed
        num_chunks = (text_length + self.max_chunk_size - 1) // self.max_chunk_size
        
        # Calculate optimal size to minimize overlap
        optimal_size = text_length // num_chunks
        
        # Ensure it's not larger than max chunk size
        return min(optimal_size, self.max_chunk_size)
    
    def validate_chunks(self, chunks: List[DocumentChunk]) -> bool:
        """
        Validate if chunks are properly formed.
        
        Args:
            chunks: List of chunks to validate
            
        Returns:
            True if all chunks are valid, False otherwise
        """
        if not chunks:
            return True
        
        for i, chunk in enumerate(chunks):
            # Check chunk index consistency
            if chunk.chunk_index != i:
                logger.error(f"Invalid chunk index: expected {i}, got {chunk.chunk_index}")
                return False
            
            # Check total chunks consistency
            if chunk.total_chunks != len(chunks):
                logger.error(f"Invalid total chunks: expected {len(chunks)}, got {chunk.total_chunks}")
                return False
            
            # Check content length
            if len(chunk.content) > self.max_chunk_size * 1.1:  # Allow 10% tolerance
                logger.error(f"Chunk {i} exceeds max size: {len(chunk.content)} > {self.max_chunk_size}")
                return False
            
            # Check position consistency
            if chunk.start_position >= chunk.end_position:
                logger.error(f"Invalid chunk positions: start={chunk.start_position}, end={chunk.end_position}")
                return False
        
        return True
    
    def _chunk_by_sentences(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk text by sentences while respecting size limits.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        # Split text into sentences
        sentences = self._split_into_sentences(text)
        
        chunks = []
        current_chunk = ""
        current_start = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from current content
                chunk = self._create_chunk(
                    content=current_chunk.strip(),
                    document_id=document_id,
                    start_pos=current_start,
                    chunk_index=len(chunks)
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = overlap_text + sentence
                current_start = text.find(overlap_text, current_start)
            else:
                current_chunk += sentence
        
        # Add final chunk if there's remaining content
        if current_chunk.strip():
            chunk = self._create_chunk(
                content=current_chunk.strip(),
                document_id=document_id,
                start_pos=current_start,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _chunk_by_characters(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk text by character count.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = min(start + self.max_chunk_size, len(text))
            
            # Extract chunk content
            content = text[start:end]
            
            # Create chunk
            chunk = self._create_chunk(
                content=content,
                document_id=document_id,
                start_pos=start,
                chunk_index=len(chunks)
            )
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.overlap_size if end < len(text) else end
        
        # Update total chunks count
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # Pattern for sentence boundaries (Chinese and English)
        sentence_pattern = r'[.!?。！？]+\s*'
        
        sentences = re.split(sentence_pattern, text)
        
        # Remove empty sentences and add back punctuation
        result = []
        for i, sentence in enumerate(sentences[:-1]):  # Exclude last empty element
            if sentence.strip():
                # Find the punctuation that was used to split
                next_start = text.find(sentence) + len(sentence)
                punct_match = re.match(sentence_pattern, text[next_start:])
                if punct_match:
                    sentence += punct_match.group()
                result.append(sentence)
        
        # Add the last sentence if it exists
        if sentences[-1].strip():
            result.append(sentences[-1])
        
        return result
    
    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from the end of current chunk.
        
        Args:
            text: Current chunk text
            
        Returns:
            Overlap text
        """
        if len(text) <= self.overlap_size:
            return text
        
        overlap = text[-self.overlap_size:]
        
        # Try to find a good break point (sentence boundary)
        if self.chunk_on_sentences:
            sentence_pattern = r'[.!?。！？]+\s*'
            matches = list(re.finditer(sentence_pattern, overlap))
            if matches:
                # Use the last sentence boundary
                last_match = matches[-1]
                overlap = overlap[last_match.end():]
        
        return overlap
    
    def _create_chunk(self, content: str, document_id: str, start_pos: int, chunk_index: int) -> DocumentChunk:
        """
        Create a DocumentChunk object.
        
        Args:
            content: Chunk content
            document_id: Document identifier
            start_pos: Start position in original text
            chunk_index: Index of this chunk
            
        Returns:
            DocumentChunk object
        """
        chunk_id = f"{document_id}_chunk_{chunk_index}"
        end_pos = start_pos + len(content)
        
        return DocumentChunk(
            document_id=document_id,
            chunk_id=chunk_id,
            content=content,
            start_position=start_pos,
            end_position=end_pos,
            chunk_index=chunk_index,
            total_chunks=0  # Will be updated later
        ) 