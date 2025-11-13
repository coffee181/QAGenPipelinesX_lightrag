"""Configuration management utilities."""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional


class ConfigManager:
    """Configuration manager for the QA Generation Pipeline."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.yaml
        """
        self.config_path = Path(config_path) if config_path else Path("config.yaml")
        self._config_data: Dict[str, Any] = {}
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    self._config_data = yaml.safe_load(f) or {}
            else:
                self._config_data = self._get_default_config()
        except Exception as e:
            raise ConfigError(f"Failed to load config from {self.config_path}: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Configuration key path (e.g., 'ocr.tesseract.lang')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self._config_data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # Handle environment variable substitution
        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var = value[2:-1]
            return os.getenv(env_var, default)
        
        return value
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set configuration value using dot notation.
        
        Args:
            key_path: Configuration key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self._config_data
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def save_config(self, config_path: Optional[Path] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            config_path: Path to save config. If None, uses current config_path
        """
        save_path = config_path or self.config_path
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config_data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            raise ConfigError(f"Failed to save config to {save_path}: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "ocr": {
                "provider": "tesseract",
                "tesseract": {
                    "lang": "chi_sim+eng",
                    "config": "--psm 6",
                    "timeout": 30,
                    "dpi": 300,
                    "enable_preprocess": True,
                    "binarize_threshold": 180,
                    "apply_median_filter": True,
                },
                "paddle": {
                    "lang": "chinese_english",  # PaddleOCR 中英混合模型
                    "use_angle_cls": True,
                    "dpi": 300,
                },
            },
            "text_chunker": {
                "max_chunk_size": 2000,
                "overlap_size": 200,
                "chunk_on_sentences": True
            },
            "question_generator": {
                "deepseek": {
                    "model": "deepseek-chat",
                    "max_tokens": 2048,
                    "temperature": 0.7,
                    "timeout": 60,
                    "questions_per_chunk": 10
                }
            },
            "rag": {
                "lightrag": {
                    "working_dir": "./lightrag_cache",
                    "llm_model": "deepseek-chat",
                    "embed_model": "text-embedding-3-small",
                    "max_context_length": 4000
                }
            },
            "file_processing": {
                "input_formats": [".pdf"],
                "output_dir": "./output",
                "temp_dir": "./temp",
                "batch_size": 10
            },
            "progress": {
                "save_interval": 5,
                "progress_file": "./progress.json"
            },
            "logging": {
                "level": "INFO",
                "format": "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
                "file": "./logs/qa_gen.log"
            }
        }


class ConfigError(Exception):
    """Custom exception for configuration errors."""
    pass 