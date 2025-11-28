"""Vectorize processed text files into the LightRAG knowledge base.

This script reuses the existing AnswerService (and therefore LightRAGImplementation)
so we rely on the exact same embedding pipeline as the rest of the application.
It supports single files, explicit file lists, or entire directories of .txt files.
The resulting vector store is persisted under the specified working directory
(working/vectorized by default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import setup_logging  # type: ignore  # noqa: E402
from src.utils.config import ConfigManager  # type: ignore  # noqa: E402
from src.utils.file_utils import FileUtils  # type: ignore  # noqa: E402
from src.implementations.lightrag_rag import LightRAGImplementation  # type: ignore  # noqa: E402
from src.services.answer_service import AnswerService  # type: ignore  # noqa: E402
from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor  # type: ignore  # noqa: E402
from src.services.progress_manager import ProgressManager  # type: ignore  # noqa: E402


DEFAULT_WORKING_DIR = (PROJECT_ROOT.parent / "working").resolve()
DEFAULT_PROCESSED_DIR = DEFAULT_WORKING_DIR / "processed"
DEFAULT_VECTOR_DIR = DEFAULT_WORKING_DIR / "vectorized"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize processed .txt documents into a LightRAG KB"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Text file or directory to vectorize (default: working/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="LightRAG working directory to store embeddings (default: working/vectorized)",
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="*",
        help="Explicit list of .txt files to vectorize (overrides --input)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config_local.yaml",
        help="Path to the pipeline configuration file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level passed to the shared logger (default: INFO)",
    )
    parser.add_argument(
        "--session-id",
        help="Optional session identifier for progress tracking",
    )
    return parser.parse_args()


def validate_text_paths(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {candidate}")
        if candidate.is_dir():
            raise ValueError(
                f"Explicit paths must refer to text files, not directories: {candidate}"
            )
        if candidate.suffix.lower() != ".txt":
            raise ValueError(f"Only .txt files are supported: {candidate}")
        resolved.append(candidate)
    return resolved


def collect_directory_documents(directory: Path) -> List[Path]:
    return sorted(p for p in directory.glob("*.txt") if p.is_file())


def insert_documents(answer_service, documents: List[Path], working_dir: Path, session_id: str | None) -> None:
    if not documents:
        raise ValueError("No text documents found to vectorize")

    for index, document_path in enumerate(documents, start=1):
        answer_service.logger.info(
            "[%d/%d] Inserting document into vector store: %s",
            index,
            len(documents),
            document_path,
        )
        answer_service.insert_documents_to_working_dir(
            document_path, working_dir, session_id
        )


def main() -> None:
    args = parse_args()

    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    # Only need answer_service for vectorization
    rag = LightRAGImplementation(config)
    markdown_processor = SimpleMarkdownProcessor()
    progress_manager = ProgressManager(config)
    answer_service = AnswerService(
        rag=rag,
        markdown_processor=markdown_processor,
        progress_manager=progress_manager,
        logger=logger
    )

    working_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(working_dir)

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths = validate_text_paths(args.paths)

    if explicit_paths:
        insert_documents(answer_service, explicit_paths, working_dir, args.session_id)
        return

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be a .txt document: {input_path}")
        insert_documents(answer_service, [input_path], working_dir, args.session_id)
    else:
        documents = collect_directory_documents(input_path)
        if not documents:
            raise ValueError(f"No .txt files found under directory: {input_path}")
        insert_documents(answer_service, documents, working_dir, args.session_id)


if __name__ == "__main__":
    main()
