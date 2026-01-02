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
        "--domain",
        help="行业/领域名称。提供后默认路径自动切换到 working/<stage>/<domain>/",
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

    def _fix_mojibake(p: Path) -> Path:
        """
        修复部分终端/环境下中文路径被“UTF-8 字节按 GBK(cp936) 误解码”导致的乱码。
        典型表现：'宝鸡机床' 变成 '瀹濋浮鏈哄簥'，从而导致 Path.exists() 失败。
        """
        s = str(p)
        # 只在“路径不存在”时尝试修复，避免误改正常路径
        try:
            fixed = s.encode("cp936", errors="strict").decode("utf-8", errors="strict")
        except Exception:
            return p
        return Path(fixed)

    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.exists():
            candidate_fixed = _fix_mojibake(candidate).expanduser().resolve()
            if candidate_fixed.exists():
                candidate = candidate_fixed
            else:
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

    if args.domain:
        domain_processed = DEFAULT_PROCESSED_DIR / args.domain
        domain_vectors = DEFAULT_VECTOR_DIR / args.domain
        domain_questions = DEFAULT_QUESTIONS_DIR / args.domain
        domain_output = DEFAULT_QA_DIR / args.domain
        if args.input == DEFAULT_PROCESSED_DIR:
            args.input = domain_processed
        if args.vectors == DEFAULT_VECTOR_DIR:
            args.vectors = domain_vectors
        if args.questions_output == DEFAULT_QUESTIONS_DIR:
            args.questions_output = domain_questions
        if args.output == DEFAULT_QA_DIR:
            args.output = domain_output

        progress_path = DEFAULT_WORKING_DIR / "progress" / args.domain / "progress.jsonl"
        config.set("progress.progress_file", str(progress_path))
        config.set("rag.lightrag.working_dir", str(domain_vectors))

    progress_file = Path(config.get("progress.progress_file", "progress.jsonl"))
    FileUtils.ensure_directory(progress_file.expanduser().resolve().parent)

    _, question_service, answer_service, progress_manager = create_services(config, logger)

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
            if progress_manager.should_skip(document_path, "qa_gen"):
                logger.info(
                    "[%d/%d] 跳过（qa_gen 已完成）：%s", idx, len(documents), document_path
                )
                continue

            logger.info(f"[{idx}/{len(documents)}] 处理文档: {document_path}")

            try:
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
                progress_manager.update_status(document_path, "qa_gen", "done")
            except Exception:
                logger.exception("生成 QA 失败：%s", document_path)
                progress_manager.update_status(document_path, "qa_gen", "failed")
                continue
    finally:
        question_service.output_dir = original_questions_output


if __name__ == "__main__":
    main()
