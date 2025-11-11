"""Markdown processor interface definition."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class MarkdownProcessorInterface(ABC):
    """Abstract interface for markdown processing implementations."""
    
    @abstractmethod
    def markdown_to_plain_text(self, markdown_text: str) -> str:
        """
        Convert markdown text to plain text.
        
        Args:
            markdown_text: Markdown formatted text
            
        Returns:
            Plain text without markdown formatting
            
        Raises:
            MarkdownProcessingError: If conversion fails
        """
        pass
    
    @abstractmethod
    def clean_llm_response(self, response: str) -> str:
        """
        Clean LLM response by removing markdown formatting.
        
        Args:
            response: Raw LLM response that may contain markdown
            
        Returns:
            Cleaned plain text response
            
        Raises:
            MarkdownProcessingError: If cleaning fails
        """
        pass
    
    @abstractmethod
    def extract_structured_content(self, markdown_text: str) -> Dict[str, Any]:
        """
        Extract structured content from markdown (headers, lists, etc.).
        
        Args:
            markdown_text: Markdown formatted text
            
        Returns:
            Dictionary containing structured content
            
        Raises:
            MarkdownProcessingError: If extraction fails
        """
        pass
    
    @abstractmethod
    def preserve_important_formatting(self, markdown_text: str) -> str:
        """
        Convert markdown to plain text while preserving important formatting.
        
        Args:
            markdown_text: Markdown formatted text
            
        Returns:
            Plain text with important formatting preserved
            
        Raises:
            MarkdownProcessingError: If processing fails
        """
        pass
    
    @abstractmethod
    def validate_markdown(self, markdown_text: str) -> bool:
        """
        Validate if text contains valid markdown syntax.
        
        Args:
            markdown_text: Text to validate
            
        Returns:
            True if valid markdown, False otherwise
        """
        pass


class MarkdownProcessingError(Exception):
    """Custom exception for markdown processing errors."""
    pass 