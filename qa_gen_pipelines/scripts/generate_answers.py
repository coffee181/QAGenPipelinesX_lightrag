"""Generate answers from existing question files and an existing LightRAG vector store.

This script runs ONLY the answer-generation stage:
  questions/*_questions.jsonl + vectorized/ -> qa-pairs/*_qa_pairs.jsonl
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
DEFAULT_QUESTIONS_DIR = DEFAULT_WORKING_DIR / "questions"
DEFAULT_VECTOR_DIR = DEFAULT_WORKING_DIR / "vectorized"
DEFAULT_QA_DIR = DEFAULT_WORKING_DIR / "qa-pairs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate answers from *_questions.jsonl files")
    parser.add_argument(
        "--domain",
        help="行业/领域名称。提供后默认路径自动切换到 working/<stage>/<domain>/",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS_DIR,
        help="Questions file or directory (default: working/questions)",
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=DEFAULT_VECTOR_DIR,
        help="LightRAG working directory (default: working/vectorized)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QA_DIR,
        help="Output directory for QA files (default: working/qa-pairs)",
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="*",
        help="Explicit list of *_questions.jsonl files (overrides --questions)",
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
        "--resume",
        action="store_true",
        help="Resume answer generation if QA files already exist",
    )
    parser.add_argument(
        "--session-id",
        help="Optional base session identifier; the script appends filenames",
    )
    return parser.parse_args()


def _fix_mojibake(p: Path) -> Path:
    """Fix common Chinese-path mojibake (UTF-8 bytes decoded as cp936)."""
    s = str(p)
    try:
        fixed = s.encode("cp936", errors="strict").decode("utf-8", errors="strict")
    except Exception:
        return p
    return Path(fixed)


def validate_question_paths(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.exists():
            candidate_fixed = _fix_mojibake(candidate).expanduser().resolve()
            if candidate_fixed.exists():
                candidate = candidate_fixed
            else:
                raise FileNotFoundError(f"Path does not exist: {candidate}")
        if candidate.is_dir():
            raise ValueError(f"Explicit paths must refer to question files, not directories: {candidate}")
        if candidate.suffix.lower() != ".jsonl":
            raise ValueError(f"Only .jsonl question files are supported: {candidate}")
        resolved.append(candidate)
    return resolved


def resolve_question_paths(questions_path: Path, explicit_paths: Optional[Iterable[Path]]) -> List[Path]:
    if explicit_paths:
        return validate_question_paths(explicit_paths)

    resolved_input = questions_path.expanduser().resolve()
    if not resolved_input.exists():
        raise FileNotFoundError(f"Questions path does not exist: {resolved_input}")

    if resolved_input.is_file():
        return [resolved_input]

    question_files = sorted(p for p in resolved_input.glob("*_questions.jsonl") if p.is_file())
    if not question_files:
        raise ValueError(f"No *_questions.jsonl files found under directory: {resolved_input}")
    return question_files


def get_session_id(base: Optional[str], stem: str) -> Optional[str]:
    if not base:
        return None
    return f"{base}_{stem}"


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    if args.domain:
        domain_questions = DEFAULT_QUESTIONS_DIR / args.domain
        domain_vectors = DEFAULT_VECTOR_DIR / args.domain
        domain_output = DEFAULT_QA_DIR / args.domain
        if args.questions == DEFAULT_QUESTIONS_DIR:
            args.questions = domain_questions
        if args.vectors == DEFAULT_VECTOR_DIR:
            args.vectors = domain_vectors
        if args.output == DEFAULT_QA_DIR:
            args.output = domain_output

        progress_path = DEFAULT_WORKING_DIR / "progress" / args.domain / "progress.jsonl"
        config.set("progress.progress_file", str(progress_path))
        config.set("rag.lightrag.working_dir", str(domain_vectors))

    progress_file = Path(config.get("progress.progress_file", "progress.jsonl"))
    FileUtils.ensure_directory(progress_file.expanduser().resolve().parent)

    _, _, answer_service, progress_manager = create_services(config, logger)

    vector_dir = args.vectors.expanduser().resolve()
    if not vector_dir.exists():
        raise FileNotFoundError(f"Vectorized working directory does not exist: {vector_dir}")

    output_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(output_dir)

    question_files = resolve_question_paths(args.questions, args.paths)
    logger.info("[answers-only] 找到 %d 个问题文件待处理", len(question_files))

    for idx, question_file in enumerate(question_files, start=1):
        # 用 question_file 反推对应的源文档名（去掉 _questions 后缀）
        base_name = question_file.stem
        if base_name.endswith("_questions"):
            base_name = base_name[: -len("_questions")]

        qa_output_file = output_dir / f"{base_name}_qa_pairs.jsonl"

        if progress_manager.should_skip(question_file, "answer_gen"):
            logger.info("[%d/%d] 跳过（answer_gen 已完成）：%s", idx, len(question_files), question_file)
            continue

        logger.info("[%d/%d] 生成答案：%s -> %s", idx, len(question_files), question_file.name, qa_output_file.name)
        try:
            session_id = get_session_id(args.session_id, base_name)
            answer_service.generate_answers_from_existing_kb(
                question_file,
                vector_dir,
                qa_output_file,
                session_id=session_id,
                resume=args.resume,
            )
            progress_manager.update_status(question_file, "answer_gen", "done")
        except Exception:
            logger.exception("[answers-only] 生成答案失败：%s", question_file)
            progress_manager.update_status(question_file, "answer_gen", "failed")
            continue


if __name__ == "__main__":
    main()


