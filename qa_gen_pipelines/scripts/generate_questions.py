"""Generate questions from processed text files.

This script runs ONLY the question-generation stage:
  processed/*.txt -> questions/*_questions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import re
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
DEFAULT_QUESTIONS_DIR = DEFAULT_WORKING_DIR / "questions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate questions from .txt documents")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Text file or directory to process (default: working/processed)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_QUESTIONS_DIR,
        help="Directory to store question files (default: working/questions)",
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
        help="Optional base session identifier; the service appends document names",
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


def _fallback_match_by_ascii_tokens(p: Path) -> Optional[Path]:
    """
    当中文路径在终端传参中发生不可逆乱码时（包含私有区字符等），尝试用 ASCII 关键片段匹配真实文件。
    例如：01-GSK 27i...20250226.txt 的 ASCII 片段仍可用于定位。
    """
    parent = p.parent
    if not parent.exists() or not parent.is_dir():
        return None

    tokens = re.findall(r"[A-Za-z0-9]+", p.name)
    if not tokens:
        return None

    candidates = [f for f in parent.glob(f"*{p.suffix}") if f.is_file()]
    hits: List[Path] = []
    for f in candidates:
        name = f.name
        if all(t.lower() in name.lower() for t in tokens):
            hits.append(f)

    if len(hits) == 1:
        return hits[0]
    return None


def validate_text_paths(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.exists():
            candidate_fixed = _fix_mojibake(candidate).expanduser().resolve()
            if candidate_fixed.exists():
                candidate = candidate_fixed
            else:
                candidate_guess = _fallback_match_by_ascii_tokens(candidate)
                if candidate_guess and candidate_guess.exists():
                    candidate = candidate_guess.resolve()
                else:
                    raise FileNotFoundError(f"Path does not exist: {candidate}")
        if candidate.is_dir():
            raise ValueError(f"Explicit paths must refer to text files, not directories: {candidate}")
        if candidate.suffix.lower() != ".txt":
            raise ValueError(f"Only .txt files are supported: {candidate}")
        resolved.append(candidate)
    return resolved


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


def main() -> None:
    args = parse_args()
    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    _, question_service, _, progress_manager = create_services(config, logger)

    output_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(output_dir)

    documents = resolve_text_paths(args.input, args.paths)
    logger.info("[questions-only] 找到 %d 个文档待处理", len(documents))

    original_questions_output = question_service.output_dir
    question_service.output_dir = output_dir
    try:
        for idx, document_path in enumerate(documents, start=1):
            if progress_manager.should_skip(document_path, "qa_gen"):
                logger.info("[%d/%d] 跳过（qa_gen 已完成）：%s", idx, len(documents), document_path)
                continue

            logger.info("[%d/%d] 生成问题：%s", idx, len(documents), document_path)
            try:
                question_service.generate_questions_from_text_file(document_path)
                progress_manager.update_status(document_path, "qa_gen", "done")
            except Exception:
                logger.exception("[questions-only] 生成问题失败：%s", document_path)
                progress_manager.update_status(document_path, "qa_gen", "failed")
                continue
    finally:
        question_service.output_dir = original_questions_output


if __name__ == "__main__":
    main()


