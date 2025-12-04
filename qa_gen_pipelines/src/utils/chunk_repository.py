"""Chunk repository for persisting and retrieving document chunks."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger

from ..models.document import DocumentChunk


@dataclass
class ChunkRecord:
    """Chunk record stored in repository."""
    chunk_id: str
    document_id: str
    content: str
    tokens: int
    chunk_order_index: int
    total_chunks: int
    start_position: int = 0
    end_position: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class ChunkRepository:
    """Repository for storing and retrieving document chunks."""
    
    def __init__(self, config: Any):
        """
        Initialize chunk repository.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.store_type = config.get("text_chunker.chunk_store.type", "json")
        self.store_path = Path(config.get("text_chunker.chunk_store.path", "./cache/chunks"))
        
        # Ensure directory exists
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        if self.store_type == "json":
            self._init_json_store()
        elif self.store_type == "sqlite":
            self._init_sqlite_store()
        else:
            raise ValueError(f"Unsupported store type: {self.store_type}")
        
        logger.info(f"ChunkRepository initialized: type={self.store_type}, path={self.store_path}")
    
    def _init_json_store(self):
        """Initialize JSON-based storage."""
        self.json_file = self.store_path / "chunks.json"
        self._chunks: Dict[str, ChunkRecord] = {}
        self._load_json()
    
    def _init_sqlite_store(self):
        """Initialize SQLite-based storage."""
        self.db_file = self.store_path / "chunks.db"
        conn = sqlite3.connect(str(self.db_file))
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                content TEXT NOT NULL,
                tokens INTEGER NOT NULL,
                chunk_order_index INTEGER NOT NULL,
                total_chunks INTEGER NOT NULL,
                start_position INTEGER DEFAULT 0,
                end_position INTEGER DEFAULT 0,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_document_id ON chunks(document_id)
        """)
        conn.commit()
        conn.close()
    
    def _load_json(self):
        """Load chunks from JSON file."""
        if self.json_file.exists():
            try:
                with open(self.json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self._chunks = {
                        k: ChunkRecord(**v) 
                        for k, v in data.items()
                    }
                logger.info(f"Loaded {len(self._chunks)} chunks from JSON")
            except Exception as e:
                logger.warning(f"Failed to load JSON chunks: {e}")
                self._chunks = {}
        else:
            self._chunks = {}
    
    def _save_json(self):
        """Save chunks to JSON file."""
        try:
            data = {
                k: asdict(v) 
                for k, v in self._chunks.items()
            }
            with open(self.json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.debug(f"Saved {len(self._chunks)} chunks to JSON")
        except Exception as e:
            logger.error(f"Failed to save JSON chunks: {e}")
    
    def upsert_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Insert or update chunks in repository.
        
        Args:
            chunks: List of DocumentChunk objects
        """
        if not chunks:
            return
        
        records = []
        for chunk in chunks:
            # Convert DocumentChunk to ChunkRecord
            record = ChunkRecord(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                content=chunk.content,
                tokens=len(chunk.content.split()),  # Simple token estimation
                chunk_order_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                start_position=chunk.start_position,
                end_position=chunk.end_position,
                metadata={}
            )
            records.append(record)
        
        if self.store_type == "json":
            for record in records:
                self._chunks[record.chunk_id] = record
            self._save_json()
        elif self.store_type == "sqlite":
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            for record in records:
                cursor.execute("""
                    INSERT OR REPLACE INTO chunks 
                    (chunk_id, document_id, content, tokens, chunk_order_index, 
                     total_chunks, start_position, end_position, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    record.chunk_id,
                    record.document_id,
                    record.content,
                    record.tokens,
                    record.chunk_order_index,
                    record.total_chunks,
                    record.start_position,
                    record.end_position,
                    json.dumps(record.metadata)
                ))
            conn.commit()
            conn.close()
        
        logger.info(f"Upserted {len(records)} chunks to repository")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[ChunkRecord]:
        """
        Get chunk by ID.
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            ChunkRecord if found, None otherwise
        """
        if self.store_type == "json":
            return self._chunks.get(chunk_id)
        elif self.store_type == "sqlite":
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, document_id, content, tokens, chunk_order_index,
                       total_chunks, start_position, end_position, metadata
                FROM chunks WHERE chunk_id = ?
            """, (chunk_id,))
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return ChunkRecord(
                    chunk_id=row[0],
                    document_id=row[1],
                    content=row[2],
                    tokens=row[3],
                    chunk_order_index=row[4],
                    total_chunks=row[5],
                    start_position=row[6],
                    end_position=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                )
            return None
        return None
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, ChunkRecord]:
        """
        Get multiple chunks by IDs.
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary mapping chunk_id to ChunkRecord
        """
        result = {}
        for chunk_id in chunk_ids:
            chunk = self.get_chunk_by_id(chunk_id)
            if chunk:
                result[chunk_id] = chunk
        return result
    
    def get_chunks_by_document(self, document_id: str) -> List[ChunkRecord]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of ChunkRecord objects, sorted by chunk_order_index
        """
        if self.store_type == "json":
            chunks = [
                chunk for chunk in self._chunks.values()
                if chunk.document_id == document_id
            ]
            return sorted(chunks, key=lambda x: x.chunk_order_index)
        elif self.store_type == "sqlite":
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT chunk_id, document_id, content, tokens, chunk_order_index,
                       total_chunks, start_position, end_position, metadata
                FROM chunks WHERE document_id = ? ORDER BY chunk_order_index
            """, (document_id,))
            rows = cursor.fetchall()
            conn.close()
            
            return [
                ChunkRecord(
                    chunk_id=row[0],
                    document_id=row[1],
                    content=row[2],
                    tokens=row[3],
                    chunk_order_index=row[4],
                    total_chunks=row[5],
                    start_position=row[6],
                    end_position=row[7],
                    metadata=json.loads(row[8]) if row[8] else {}
                )
                for row in rows
            ]
        return []
    
    def delete_chunks_by_document(self, document_id: str) -> int:
        """
        Delete all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            Number of deleted chunks
        """
        if self.store_type == "json":
            count = sum(1 for chunk_id in list(self._chunks.keys())
                       if self._chunks[chunk_id].document_id == document_id)
            self._chunks = {
                k: v for k, v in self._chunks.items()
                if v.document_id != document_id
            }
            self._save_json()
            return count
        elif self.store_type == "sqlite":
            conn = sqlite3.connect(str(self.db_file))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
            count = cursor.rowcount
            conn.commit()
            conn.close()
            return count
        return 0
    
    def get_content_by_id(self, chunk_id: str) -> Optional[str]:
        """
        Get chunk content by ID (convenience method).
        
        Args:
            chunk_id: Chunk ID
            
        Returns:
            Chunk content if found, None otherwise
        """
        chunk = self.get_chunk_by_id(chunk_id)
        return chunk.content if chunk else None
    
    def get_contents_by_ids(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Get chunk contents by IDs (convenience method).
        
        Args:
            chunk_ids: List of chunk IDs
            
        Returns:
            Dictionary mapping chunk_id to content
        """
        chunks = self.get_chunks_by_ids(chunk_ids)
        return {chunk_id: chunk.content for chunk_id, chunk in chunks.items()}

