"""Simple markdown processor implementation."""

import re
from typing import Dict, Any
from loguru import logger
import markdown

from ..interfaces.markdown_processor_interface import MarkdownProcessorInterface, MarkdownProcessingError


class SimpleMarkdownProcessor(MarkdownProcessorInterface):
    """Simple markdown processing implementation."""
    
    def __init__(self):
        """Initialize markdown processor."""
        self.md = markdown.Markdown(extensions=['extra', 'codehilite'])
        logger.info("Simple markdown processor initialized")
    
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
        try:
            if not markdown_text:
                return ""
            
            # Remove markdown formatting patterns
            text = self._remove_markdown_formatting(markdown_text)
            
            # Clean up extra whitespace
            text = self._clean_whitespace(text)
            
            return text
            
        except Exception as e:
            raise MarkdownProcessingError(f"Failed to convert markdown to plain text: {e}")
    
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
        try:
            if not response:
                return ""
            
            text = response
            
            # Remove DeepSeek R1 <think> tags and content
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove References section and similar patterns - more aggressive
            text = re.sub(r'\n\*\*References\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\nReferences.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n\*\*参考资料\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n参考资料.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n\*\*参考.*\*\*.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\n参考.*$', '', text, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove reference patterns more thoroughly
            text = re.sub(r'\n\[DC\d+\].*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n\[KG\d+\].*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n见\[DC\d+\].*$', '', text, flags=re.MULTILINE)
            text = re.sub(r'\n见\[KG\d+\].*$', '', text, flags=re.MULTILINE)
            
            # Remove markdown headers more thoroughly
            text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
            
            # Remove all markdown bold/italic formatting
            text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text, flags=re.DOTALL)  # Bold italic
            text = re.sub(r'\*\*(.*?)\*\*', r'\1', text, flags=re.DOTALL)  # Bold
            text = re.sub(r'\*(.*?)\*', r'\1', text, flags=re.DOTALL)  # Italic
            text = re.sub(r'___(.*?)___', r'\1', text, flags=re.DOTALL)  # Bold italic
            text = re.sub(r'__(.*?)__', r'\1', text, flags=re.DOTALL)  # Bold
            text = re.sub(r'_(.*?)_', r'\1', text, flags=re.DOTALL)  # Italic
            
            # Remove code blocks and inline code
            text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)
            text = re.sub(r'`([^`]*)`', r'\1', text)
            
            # Remove links and images
            text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
            text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)
            
            # Remove list markers
            text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
            
            # Remove blockquotes
            text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
            
            # Remove horizontal rules
            text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)
            
            # Remove table formatting
            text = re.sub(r'\|', ' ', text)
            text = re.sub(r'^[-\s|:]+$', '', text, flags=re.MULTILINE)
            
            # Clean up whitespace more aggressively
            text = re.sub(r' +', ' ', text)  # Multiple spaces to single
            text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines
            text = re.sub(r'^\s+', '', text, flags=re.MULTILINE)  # Leading spaces
            text = re.sub(r'\s+$', '', text, flags=re.MULTILINE)  # Trailing spaces
            
            # Remove empty lines
            lines = [line for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            # Clean up excessive newlines and line breaks
            text = self._clean_excessive_newlines(text)
            
            # Additional LLM artifact cleaning
            text = self._clean_llm_artifacts(text)
            
            return text.strip()
            
        except Exception as e:
            raise MarkdownProcessingError(f"Failed to clean LLM response: {e}")
    
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
        try:
            structure = {
                "headers": [],
                "lists": [],
                "code_blocks": [],
                "links": [],
                "images": [],
                "tables": [],
                "plain_text": ""
            }
            
            # Extract headers
            structure["headers"] = self._extract_headers(markdown_text)
            
            # Extract lists
            structure["lists"] = self._extract_lists(markdown_text)
            
            # Extract code blocks
            structure["code_blocks"] = self._extract_code_blocks(markdown_text)
            
            # Extract links
            structure["links"] = self._extract_links(markdown_text)
            
            # Extract images
            structure["images"] = self._extract_images(markdown_text)
            
            # Extract tables
            structure["tables"] = self._extract_tables(markdown_text)
            
            # Get plain text
            structure["plain_text"] = self.markdown_to_plain_text(markdown_text)
            
            return structure
            
        except Exception as e:
            raise MarkdownProcessingError(f"Failed to extract structured content: {e}")
    
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
        try:
            if not markdown_text:
                return ""
            
            text = markdown_text
            
            # Preserve headers by converting to numbered format
            text = self._preserve_headers(text)
            
            # Preserve lists by converting to simple format
            text = self._preserve_lists(text)
            
            # Preserve emphasis by converting to uppercase/brackets
            text = self._preserve_emphasis(text)
            
            # Remove other markdown formatting
            text = self._remove_other_formatting(text)
            
            # Clean up whitespace
            text = self._clean_whitespace(text)
            
            return text
            
        except Exception as e:
            raise MarkdownProcessingError(f"Failed to preserve formatting: {e}")
    
    def validate_markdown(self, markdown_text: str) -> bool:
        """
        Validate if text contains valid markdown syntax.
        
        Args:
            markdown_text: Text to validate
            
        Returns:
            True if valid markdown, False otherwise
        """
        if not markdown_text:
            return False
        
        # Check for common markdown patterns
        markdown_patterns = [
            r'^#{1,6}\s',  # Headers
            r'\*\*.*?\*\*',  # Bold
            r'\*.*?\*',  # Italic
            r'`.*?`',  # Inline code
            r'```.*?```',  # Code blocks
            r'\[.*?\]\(.*?\)',  # Links
            r'!\[.*?\]\(.*?\)',  # Images
            r'^\s*[-*+]\s',  # Lists
            r'^\s*\d+\.\s',  # Numbered lists
            r'^\|.*\|$',  # Tables
        ]
        
        for pattern in markdown_patterns:
            if re.search(pattern, markdown_text, re.MULTILINE | re.DOTALL):
                return True
        
        return False
    
    def _remove_markdown_formatting(self, text: str) -> str:
        """Remove markdown formatting patterns."""
        # Remove headers
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
        
        # Remove bold and italic
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # Bold italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'___(.*?)___', r'\1', text)  # Bold italic
        text = re.sub(r'__(.*?)__', r'\1', text)  # Bold
        text = re.sub(r'_(.*?)_', r'\1', text)  # Italic
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
        
        # Remove links
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        
        # Remove images
        text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)
        
        # Remove horizontal rules
        text = re.sub(r'^[-*_]{3,}$', '', text, flags=re.MULTILINE)
        
        # Remove blockquotes
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)
        
        # Remove list markers
        text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
        
        # Remove table formatting
        text = re.sub(r'\|', ' ', text)
        text = re.sub(r'^[-\s|:]+$', '', text, flags=re.MULTILINE)
        
        return text
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up extra whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        
        # Remove leading/trailing whitespace from lines
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove empty lines at start and end
        text = text.strip()
        
        return text
    
    def _clean_basic_formatting(self, text: str) -> str:
        """Clean basic formatting from non-markdown text."""
        # Remove common formatting artifacts
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Code
        
        return self._clean_whitespace(text)
    
    def _clean_excessive_newlines(self, text: str) -> str:
        """Clean excessive newlines and line breaks."""
        # Remove standalone \n characters that appear as literal text
        text = re.sub(r'\\n', ' ', text)  # Replace \n with space
        
        # Remove excessive newlines (more than 2 consecutive)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Clean up patterns where newlines appear mid-sentence
        # Replace newline between Chinese characters with space
        text = re.sub(r'([\u4e00-\u9fff])\n([\u4e00-\u9fff])', r'\1\2', text)
        
        # Replace newline between words with space (but preserve paragraph breaks)
        text = re.sub(r'([^\n])\n([^\n])', r'\1 \2', text)
        
        # Clean up multiple spaces that may have been created
        text = re.sub(r' +', ' ', text)
        
        # Restore proper paragraph breaks (double newlines)
        text = re.sub(r'([。！？])\s+([^\n])', r'\1\n\n\2', text)  # After Chinese punctuation
        text = re.sub(r'([.!?])\s+([A-Z])', r'\1\n\n\2', text)  # After English punctuation
        
        return text.strip()
    
    def _clean_llm_artifacts(self, text: str) -> str:
        """Clean LLM-specific artifacts."""
        # Remove DeepSeek R1 <think> tags and content (additional cleanup)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove common LLM response patterns
        text = re.sub(r'^(答案?[:：]?\s*)', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(回答[:：]?\s*)', '', text, flags=re.MULTILINE)
        text = re.sub(r'^(解答[:：]?\s*)', '', text, flags=re.MULTILINE)
        
        # Remove JSON-like artifacts that might appear
        text = re.sub(r'\{"role":\s*"[^"]*",\s*"content":\s*"[^"]*"\}', '', text)
        
        # Remove reference patterns
        text = re.sub(r'\[DC\d+\]', '', text)  # Remove [DC1], [DC2], etc.
        text = re.sub(r'见\[DC\d+\]', '', text)  # Remove Chinese references
        text = re.sub(r'\[KG\d+\]', '', text)  # Remove [KG1], [KG2], etc.
        text = re.sub(r'见\[KG\d+\]', '', text)  # Remove Chinese KG references
        
        # Remove additional reference patterns - more aggressive
        text = re.sub(r'参考资料[:：]?.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'参考[:：]?.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        text = re.sub(r'References[:：]?.*$', '', text, flags=re.MULTILINE | re.IGNORECASE)
        
        # Remove any remaining reference lines
        lines = text.split('\n')
        cleaned_lines = []
        skip_reference = False
        
        for line in lines:
            line = line.strip()
            # Skip lines that start with reference patterns
            if (line.startswith('[DC') or line.startswith('[KG') or 
                line.startswith('见[DC') or line.startswith('见[KG') or
                line.startswith('参考资料') or line.startswith('参考') or
                line.startswith('References')):
                skip_reference = True
                continue
            # Skip empty lines after references
            if skip_reference and not line:
                continue
            # Reset skip flag when we encounter non-empty content
            if line:
                skip_reference = False
            cleaned_lines.append(line)
        
        text = '\n'.join(cleaned_lines)
        
        return text
    
    def _extract_headers(self, text: str) -> list:
        """Extract headers from markdown."""
        headers = []
        pattern = r'^(#{1,6})\s*(.*?)$'
        
        for match in re.finditer(pattern, text, re.MULTILINE):
            level = len(match.group(1))
            content = match.group(2).strip()
            headers.append({"level": level, "content": content})
        
        return headers
    
    def _extract_lists(self, text: str) -> list:
        """Extract lists from markdown."""
        lists = []
        
        # Unordered lists
        unordered_pattern = r'^\s*[-*+]\s*(.*?)$'
        for match in re.finditer(unordered_pattern, text, re.MULTILINE):
            lists.append({"type": "unordered", "content": match.group(1).strip()})
        
        # Ordered lists
        ordered_pattern = r'^\s*\d+\.\s*(.*?)$'
        for match in re.finditer(ordered_pattern, text, re.MULTILINE):
            lists.append({"type": "ordered", "content": match.group(1).strip()})
        
        return lists
    
    def _extract_code_blocks(self, text: str) -> list:
        """Extract code blocks from markdown."""
        code_blocks = []
        
        # Fenced code blocks
        pattern = r'```(\w*)\n?(.*?)```'
        for match in re.finditer(pattern, text, re.DOTALL):
            language = match.group(1) or "text"
            code = match.group(2).strip()
            code_blocks.append({"language": language, "code": code})
        
        return code_blocks
    
    def _extract_links(self, text: str) -> list:
        """Extract links from markdown."""
        links = []
        pattern = r'\[([^\]]*)\]\(([^\)]*)\)'
        
        for match in re.finditer(pattern, text):
            text_content = match.group(1)
            url = match.group(2)
            links.append({"text": text_content, "url": url})
        
        return links
    
    def _extract_images(self, text: str) -> list:
        """Extract images from markdown."""
        images = []
        pattern = r'!\[([^\]]*)\]\(([^\)]*)\)'
        
        for match in re.finditer(pattern, text):
            alt_text = match.group(1)
            url = match.group(2)
            images.append({"alt": alt_text, "url": url})
        
        return images
    
    def _extract_tables(self, text: str) -> list:
        """Extract tables from markdown."""
        tables = []
        
        # Simple table detection
        lines = text.split('\n')
        in_table = False
        current_table = []
        
        for line in lines:
            if '|' in line and line.strip():
                if not in_table:
                    in_table = True
                    current_table = []
                current_table.append(line.strip())
            else:
                if in_table and current_table:
                    tables.append({"rows": current_table})
                    current_table = []
                in_table = False
        
        # Add last table if exists
        if in_table and current_table:
            tables.append({"rows": current_table})
        
        return tables
    
    def _preserve_headers(self, text: str) -> str:
        """Preserve headers in plain text format."""
        def replace_header(match):
            level = len(match.group(1))
            content = match.group(2).strip()
            return f"{'=' * level} {content} {'=' * level}"
        
        return re.sub(r'^(#{1,6})\s*(.*?)$', replace_header, text, flags=re.MULTILINE)
    
    def _preserve_lists(self, text: str) -> str:
        """Preserve lists in plain text format."""
        # Keep list markers as is for now
        return text
    
    def _preserve_emphasis(self, text: str) -> str:
        """Preserve emphasis in plain text format."""
        # Convert bold to uppercase
        text = re.sub(r'\*\*(.*?)\*\*', lambda m: m.group(1).upper(), text)
        text = re.sub(r'__(.*?)__', lambda m: m.group(1).upper(), text)
        
        # Convert italic to brackets
        text = re.sub(r'\*(.*?)\*', r'[\1]', text)
        text = re.sub(r'_(.*?)_', r'[\1]', text)
        
        return text
    
    def _remove_other_formatting(self, text: str) -> str:
        """Remove other markdown formatting."""
        # Remove remaining markdown syntax
        text = re.sub(r'`(.*?)`', r'\1', text)  # Inline code
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # Code blocks
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # Links
        text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # Images
        
        return text 