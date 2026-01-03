"""Simple text chunker implementation."""

import re
from typing import List, Optional
from loguru import logger

from ..interfaces.text_chunker_interface import TextChunkerInterface, ChunkingError
from ..models.document import Document, DocumentChunk
from ..utils.config import ConfigManager
from ..utils.lightrag_utils import compute_lightrag_chunk_id

# Try to import LightRAG chunking function
try:
    from lightrag.operate import chunking_by_token_size
    # 尝试导入tokenizer，如果失败则使用tiktoken
    try:
        from lightrag.llm import TiktokenTokenizer
        LIGHTRAG_TOKENIZER_AVAILABLE = True
    except ImportError:
        try:
            import tiktoken
            TiktokenTokenizer = tiktoken.Encoding
            LIGHTRAG_TOKENIZER_AVAILABLE = True
            logger.info("使用tiktoken作为tokenizer")
        except ImportError:
            LIGHTRAG_TOKENIZER_AVAILABLE = False
            logger.warning("tiktoken不可用，将使用字符切分")
    
    LIGHTRAG_AVAILABLE = True
    logger.info("LightRAG chunking功能可用")
except ImportError:
    LIGHTRAG_AVAILABLE = False
    LIGHTRAG_TOKENIZER_AVAILABLE = False
    logger.warning("LightRAG not available, falling back to character-based chunking")


class SimpleTextChunker(TextChunkerInterface):
    """Simple text chunking implementation."""
    
    def __init__(self, config: ConfigManager, chunk_repository=None):
        """
        Initialize text chunker.
        
        Args:
            config: Configuration object
            chunk_repository: Optional ChunkRepository for persisting chunks
        """
        self.config = config
        # 本地 chunk 持久化关闭
        self.chunk_repository = None
        
        # 🚀 优化：支持 token 级切分（与 LightRAG 一致）
        self.use_token_chunking = config.get("text_chunker.use_token_chunking", False)
        
        if self.use_token_chunking and LIGHTRAG_AVAILABLE and LIGHTRAG_TOKENIZER_AVAILABLE:
            # Token 级切分配置
            self.tokenizer_model = config.get("text_chunker.tokenizer_model", "cl100k_base")
            self.chunk_token_size = config.get("text_chunker.chunk_token_size", 1200)
            self.chunk_overlap_token_size = config.get("text_chunker.chunk_overlap_token_size", 100)
            
            # Initialize tokenizer
            try:
                if TiktokenTokenizer == tiktoken.Encoding:
                    # 使用tiktoken
                    self.tokenizer = tiktoken.get_encoding(self.tokenizer_model)
                else:
                    # 使用LightRAG的tokenizer
                    self.tokenizer = TiktokenTokenizer(model_name=self.tokenizer_model)
                logger.info(f"🔧 Token切分器初始化完成: model={self.tokenizer_model}, "
                          f"chunk_size={self.chunk_token_size} tokens, "
                          f"overlap={self.chunk_overlap_token_size} tokens")
            except Exception as e:
                logger.warning(f"Failed to initialize tokenizer: {e}, falling back to character chunking")
                self.use_token_chunking = False
        else:
            self.use_token_chunking = False
        
        # 字符级切分配置（向后兼容）
        if not self.use_token_chunking:
            self.max_chunk_size = config.get("text_chunker.max_chunk_size", 2000)
            self.overlap_size = config.get("text_chunker.overlap_size", 200)
            self.chunk_on_sentences = config.get("text_chunker.chunk_on_sentences", True)
            logger.info(f"Character chunker initialized: max_size={self.max_chunk_size}, "
                      f"overlap={self.overlap_size}, sentences={self.chunk_on_sentences}")
        
        # Chunk 持久化关闭
        self.persist_chunks = False
    
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
            
            logger.info(f" 开始文档切分: {document_id} (token切分={self.use_token_chunking})")
            
            # 优化：优先使用 token 级切分（与 LightRAG 一致）
            if self.use_token_chunking:
                chunks = self._chunk_by_tokens(text, document_id)
            elif self.chunk_on_sentences:
                logger.info(f" 使用句子级切分")
                chunks = self._chunk_by_sentences(text, document_id)
            else:
                logger.info(f" 使用字符级切分")
                chunks = self._chunk_by_characters(text, document_id)
            
            # 优化：如果配置了持久化，保存到 ChunkRepository
            if self.persist_chunks and chunks:
                try:
                    self.chunk_repository.upsert_chunks(chunks)
                    logger.info(f" 已持久化 {len(chunks)} 个chunks到仓库")
                except Exception as e:
                    logger.warning(f"  Chunk持久化失败: {e}")
            
            logger.info(f" 文档切分完成: {document_id} → {len(chunks)}个chunks")
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
                overlap = overlap[:last_match.start()]
        
        return overlap
    
    def _chunk_by_tokens(self, text: str, document_id: str) -> List[DocumentChunk]:
        """
        Chunk text using token-based chunking with LightRAG compatibility.
        
        Args:
            text: Text to chunk
            document_id: Document identifier
            
        Returns:
            List of DocumentChunk objects
        """
        logger.info(f"🔧 开始Token切分: document={document_id}, text_length={len(text)}字符")
        
        # Use LightRAG's chunking function
        chunk_dicts = chunking_by_token_size(
            tokenizer=self.tokenizer,
            content=text,
            split_by_character=None,
            split_by_character_only=False,
            overlap_token_size=self.chunk_overlap_token_size,
            max_token_size=self.chunk_token_size
        )
        
        logger.info(f"🔧 LightRAG切分完成: 生成 {len(chunk_dicts)} 个原始chunks")
        
        chunks = []
        current_pos = 0
        total_tokens = 0
        
        for idx, chunk_dict in enumerate(chunk_dicts):
            content = chunk_dict.get("content", "").strip()
            if not content:
                continue
                
            # 计算token数量
            tokens = len(self.tokenizer.encode(content))
            total_tokens += tokens
            
            # 🚀 优化：使用 LightRAG 的 chunk_id 计算方式
            lightrag_chunk_id = compute_lightrag_chunk_id(content)
            if not lightrag_chunk_id:
                # Fallback to document-based ID if computation fails
                lightrag_chunk_id = f"{document_id}_chunk_{idx}"
            
            # Find position in original text
            start_pos = text.find(content, current_pos)
            end_pos = start_pos + len(content)
            current_pos = start_pos
            
            # 🚀 优化：使用 LightRAG 兼容的 chunk_id
            chunk = DocumentChunk(
                document_id=document_id,
                chunk_id=lightrag_chunk_id,
                content=content,
                start_position=start_pos,
                end_position=end_pos,
                chunk_index=chunk_dict.get("chunk_order_index", idx),
                total_chunks=len(chunk_dicts)
            )
            chunks.append(chunk)
            
            # 详细日志：每个chunk的信息
            logger.debug(f"🔧 Chunk {idx+1}/{len(chunk_dicts)}: "
                        f"tokens={tokens}, chars={len(content)}, "
                        f"id={lightrag_chunk_id[:12]}..., "
                        f"pos={start_pos}-{end_pos}")
        
        # 更新总chunk数
        for chunk in chunks:
            chunk.total_chunks = len(chunks)
        
        # 总结日志
        avg_tokens = total_tokens // len(chunks) if chunks else 0
        logger.info(f"🔧 Token切分完成: {len(chunks)}个chunks, "
                   f"总tokens={total_tokens}, 平均={avg_tokens}tokens/chunk")
        
        return chunks
    
    def _create_chunk(self, content: str, document_id: str, start_pos: int, chunk_index: int, 
                     use_lightrag_id: bool = False) -> DocumentChunk:
        """
        Create a DocumentChunk object.
        
        Args:
            content: Chunk content
            document_id: Document identifier
            start_pos: Start position in original text
            chunk_index: Index of this chunk
            use_lightrag_id: Whether to use LightRAG-compatible chunk_id
            
        Returns:
            DocumentChunk object
        """
        if use_lightrag_id:
            # 🚀 优化：使用 LightRAG 的 chunk_id 计算方式
            chunk_id = compute_lightrag_chunk_id(content)
            if not chunk_id:
                chunk_id = f"{document_id}_chunk_{chunk_index}"
        else:
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