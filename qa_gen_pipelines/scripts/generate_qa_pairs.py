"""Generate QA pairs by running question+answer pipeline on processed texts.

This script orchestrates question generation and answer generation on top of the
existing service layer. It assumes that documents have already been OCR'd into
plain text (e.g., via preprocess_pdfs.py) and that a vector store already
exists under working/vectorized (created via vectorize_texts.py).

The workflow:
1. Generate questions from text files → working/questions/*.jsonl
2. Generate answers from questions + vector store → working/qa-pairs/*.jsonl

The produced QA pairs are written to working/qa-pairs by default.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import create_services, setup_logging  # type: ignore  # noqa: E402
from src.utils.config import ConfigManager  # type: ignore  # noqa: E402
from src.utils.file_utils import FileUtils  # type: ignore  # noqa: E402


DEFAULT_WORKING_DIR = (PROJECT_ROOT.parent / "working").resolve()
DEFAULT_PROCESSED_DIR = DEFAULT_WORKING_DIR / "processed"
DEFAULT_VECTOR_DIR = DEFAULT_WORKING_DIR / "vectorized"
DEFAULT_QUESTIONS_DIR = DEFAULT_WORKING_DIR / "questions"
DEFAULT_QA_DIR = DEFAULT_WORKING_DIR / "qa-pairs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from processed text documents"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Processed text file or directory (default: working/processed)",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="LightRAG working directory containing vectorized data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QA_DIR,
        help="Directory to store QA pair JSONL files (default: working/qa-pairs)",
    )
    parser.add_argument(
        "--questions-output",
        type=Path,
        default=DEFAULT_QUESTIONS_DIR,
        help="Directory to store intermediate question files (default: working/questions)",
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="*",
        help="Explicit list of text files to process (overrides --input)",
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
        help="Optional base session identifier; the script appends document names",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume answer generation if QA files already exist",
    )
    return parser.parse_args()


def resolve_text_paths(input_path: Path, explicit_paths: Optional[Iterable[Path]]) -> List[Path]:
    if explicit_paths:
        return validate_text_paths(explicit_paths)

    resolved_input = input_path.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Input path does not exist: {resolved_input}")

    if resolved_input.is_file():
        if resolved_input.suffix.lower() != ".txt":
            raise ValueError(f"Input file must be a .txt document: {resolved_input}")
        return [resolved_input]

    documents = sorted(p for p in resolved_input.glob("*.txt") if p.is_file())
    if not documents:
        raise ValueError(f"No .txt files found under directory: {resolved_input}")
    return documents


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


def get_session_id(base: Optional[str], document_stem: str) -> Optional[str]:
    if not base:
        return None
    return f"{base}_{document_stem}"


def main() -> None:
    args = parse_args()

    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    _, question_service, answer_service, _ = create_services(config, logger)

    output_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(output_dir)

    questions_dir = args.questions_output.expanduser().resolve()
    FileUtils.ensure_directory(questions_dir)

    vector_dir = args.vectors.expanduser().resolve()
    if not vector_dir.exists():
        raise FileNotFoundError(
            f"Vectorized working directory does not exist: {vector_dir}"
        )

    logger.info("开始QA生成流程")
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"向量库: {vector_dir}")
    logger.info(f"问题目录: {questions_dir}")

    documents = resolve_text_paths(args.input, args.paths)

    logger.info(f"找到 {len(documents)} 个文档待处理")

    original_questions_output = question_service.output_dir
    question_service.output_dir = questions_dir

    try:
        for idx, document_path in enumerate(documents, start=1):
            logger.info(f"[{idx}/{len(documents)}] 处理文档: {document_path}")

            logger.info("步骤1: 生成问题")
            question_service.generate_questions_from_text_file(document_path)

            question_file = questions_dir / f"{document_path.stem}_questions.jsonl"
            if not question_file.exists():
                raise FileNotFoundError(
                    f"Expected questions file not found: {question_file}"
                )

            qa_output_file = output_dir / f"{document_path.stem}_qa_pairs.jsonl"
            session_id = get_session_id(args.session_id, document_path.stem)

            logger.info("步骤2: 生成答案")
            logger.info(f"使用向量库: {vector_dir}")
            answer_service.generate_answers_from_existing_kb(
                question_file,
                vector_dir,
                qa_output_file,
                session_id=session_id,
                resume=args.resume,
            )
    finally:
        question_service.output_dir = original_questions_output


if __name__ == "__main__":
    main()
