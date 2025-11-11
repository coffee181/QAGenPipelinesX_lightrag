"""File handling utilities."""

import json
import jsonlines
from pathlib import Path
from typing import List, Dict, Any, Optional, Generator
from loguru import logger


class FileUtils:
    """Utility class for file operations."""
    
    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """
        Ensure directory exists, create if not.
        
        Args:
            directory: Directory path to ensure
        """
        directory.mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def get_files_by_extension(directory: Path, extensions: List[str]) -> List[Path]:
        """
        Get all files with specified extensions from directory.
        
        Args:
            directory: Directory to search
            extensions: List of file extensions (e.g., ['.pdf', '.txt'])
            
        Returns:
            List of file paths
        """
        files = []
        for ext in extensions:
            files.extend(directory.glob(f"*{ext}"))
        return sorted(files)
    
    @staticmethod
    def save_text_file(content: str, file_path: Path) -> None:
        """
        Save text content to file.
        
        Args:
            content: Text content to save
            file_path: Path to save file
        """
        FileUtils.ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        logger.info(f"Saved text file: {file_path}")
    
    @staticmethod
    def load_text_file(file_path: Path) -> str:
        """
        Load text content from file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Text content
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    @staticmethod
    def save_json_file(data: Dict[str, Any], file_path: Path) -> None:
        """
        Save data to JSON file.
        
        Args:
            data: Data to save
            file_path: Path to save file
        """
        FileUtils.ensure_directory(file_path.parent)
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved JSON file: {file_path}")
    
    @staticmethod
    def load_json_file(file_path: Path) -> Dict[str, Any]:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            Loaded data
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def append_to_jsonl(data: Dict[str, Any], file_path: Path) -> None:
        """
        Append data to JSONL file.
        
        Args:
            data: Data to append
            file_path: Path to JSONL file
        """
        FileUtils.ensure_directory(file_path.parent)
        with jsonlines.open(file_path, mode='a') as writer:
            writer.write(data)
    
    @staticmethod
    def read_jsonl_file(file_path: Path) -> Generator[Dict[str, Any], None, None]:
        """
        Read data from JSONL file.
        
        Args:
            file_path: Path to JSONL file
            
        Yields:
            Data objects from file
        """
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                yield obj
    
    @staticmethod
    def save_jsonl_file(data_list: List[Dict[str, Any]], file_path: Path) -> None:
        """
        Save list of data to JSONL file.
        
        Args:
            data_list: List of data objects
            file_path: Path to save file
        """
        FileUtils.ensure_directory(file_path.parent)
        with jsonlines.open(file_path, mode='w') as writer:
            writer.write_all(data_list)
        logger.info(f"Saved JSONL file: {file_path}")
    
    @staticmethod
    def load_jsonl(file_path: Path) -> List[Dict[str, Any]]:
        """
        Load all data from JSONL file as a list.
        
        Args:
            file_path: Path to JSONL file
            
        Returns:
            List of data objects from file
        """
        data = []
        with jsonlines.open(file_path, mode='r') as reader:
            for obj in reader:
                data.append(obj)
        return data
    
    @staticmethod
    def file_exists(file_path: Path) -> bool:
        """
        Check if file exists.
        
        Args:
            file_path: Path to check
            
        Returns:
            True if file exists, False otherwise
        """
        return file_path.exists() and file_path.is_file()
    
    @staticmethod
    def get_file_size(file_path: Path) -> int:
        """
        Get file size in bytes.
        
        Args:
            file_path: Path to file
            
        Returns:
            File size in bytes
        """
        return file_path.stat().st_size if file_path.exists() else 0
    
    @staticmethod
    def create_backup(file_path: Path, backup_suffix: str = ".bak") -> Path:
        """
        Create backup of file.
        
        Args:
            file_path: Path to file to backup
            backup_suffix: Suffix for backup file
            
        Returns:
            Path to backup file
        """
        backup_path = file_path.with_suffix(file_path.suffix + backup_suffix)
        if file_path.exists():
            backup_path.write_bytes(file_path.read_bytes())
            logger.info(f"Created backup: {backup_path}")
        return backup_path 