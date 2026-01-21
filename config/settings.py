"""Pydantic 配置加载器"""

from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class PathsConfig(BaseModel):
    """路径配置"""
    working_dir: str = "./working"
    raw_dir: str = "raw"
    processed_dir: str = "processed"
    lightrag_db_dir: str = "lightrag_db"
    output_dir: str = "output"
    progress_file: str = "progress/progress.jsonl"

    def get_absolute_paths(self, base_dir: Optional[Path] = None) -> dict:
        """获取绝对路径"""
        base = Path(base_dir) if base_dir else Path.cwd()
        working = (base / self.working_dir).resolve()
        return {
            "working_dir": working,
            "raw_dir": working / self.raw_dir,
            "processed_dir": working / self.processed_dir,
            "lightrag_db_dir": working / self.lightrag_db_dir,
            "output_dir": working / self.output_dir,
            "progress_file": working / self.progress_file,
        }


class OCRConfig(BaseModel):
    """OCR配置"""
    lang: str = "ch"
    use_angle_cls: bool = True
    dpi: int = 300


class ChunkingConfig(BaseModel):
    """文本分块配置"""
    use_token_chunking: bool = True
    tokenizer_model: str = "cl100k_base"
    chunk_token_size: int = 1200
    chunk_overlap_token_size: int = 100


class EmbeddingConfig(BaseModel):
    """Embedding配置"""
    provider: str = "ollama"
    model: str = "bge-m3"
    dim: int = 1024
    base_url: str = "http://localhost:11434"
    timeout: int = 1200
    max_retries: int = 3


class LLMConfig(BaseModel):
    """LLM配置"""
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:32b"
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 1800
    max_retries: int = 5


class QueryConfig(BaseModel):
    """查询配置"""
    top_k: int = 20
    chunk_top_k: int = 10
    max_entity_tokens: int = 10000
    max_relation_tokens: int = 10000
    max_total_tokens: int = 40000
    cosine_threshold: float = 0.2
    related_chunk_number: int = 2


class LightRAGConfig(BaseModel):
    """LightRAG配置"""
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    query: QueryConfig = Field(default_factory=QueryConfig)


class KGContextConfig(BaseModel):
    """知识图谱上下文配置"""
    enabled: bool = True
    max_entities: int = 5
    max_relations: int = 5
    max_snippets: int = 2
    snippet_chars: int = 200
    max_related_chunk_ids: int = 10


class QualityConfig(BaseModel):
    """问题质量控制配置"""
    enable_deduplication: bool = True
    dedup_similarity_threshold: float = 0.85
    enable_quality_filter: bool = True


class QuestionGenLLMConfig(BaseModel):
    """问题生成LLM配置"""
    base_url: str = "http://localhost:11434"
    model: str = "deepseek-r1:32b"
    max_tokens: int = 2048
    temperature: float = 0.7
    timeout: int = 30000


class QuestionGenConfig(BaseModel):
    """问题生成配置"""
    llm: QuestionGenLLMConfig = Field(default_factory=QuestionGenLLMConfig)
    questions_per_chunk: int = 10
    kg_context: KGContextConfig = Field(default_factory=KGContextConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)


class PromptsConfig(BaseModel):
    """Prompts配置"""
    system_prompt: str = ""
    human_prompt: str = ""


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"


class Settings(BaseModel):
    """全局配置"""
    paths: PathsConfig = Field(default_factory=PathsConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    lightrag: LightRAGConfig = Field(default_factory=LightRAGConfig)
    question_gen: QuestionGenConfig = Field(default_factory=QuestionGenConfig)
    prompts: PromptsConfig = Field(default_factory=PromptsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Settings":
        """
        从YAML文件加载配置
        
        Args:
            config_path: 配置文件路径，默认为 config/config.yaml
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        if not config_path.exists():
            return cls()
        
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        
        return cls(**data)

    def get_working_paths(self, base_dir: Optional[Path] = None) -> dict:
        """获取所有工作路径的绝对路径"""
        return self.paths.get_absolute_paths(base_dir)


def load_settings(config_path: Optional[Path] = None) -> Settings:
    """加载配置的便捷函数"""
    return Settings.load(config_path)

