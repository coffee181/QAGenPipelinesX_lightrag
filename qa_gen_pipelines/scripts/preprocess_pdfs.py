"""Command-line helper to preprocess PDF files into text documents.

This script wraps the existing PDFProcessor service so that users can choose to
process a single PDF, multiple explicit PDFs, or every PDF within a directory.
The extracted text results are written to the configured processed directory
(working/processed by default).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, List, Optional

try:
    import fitz  # type: ignore  # PyMuPDF
except ImportError:  # pragma: no cover - 环境缺少依赖时提示
    fitz = None

# Ensure the project root (qa_gen_pipelines) is on PYTHONPATH so we can reuse
# the service layer that already exists inside the application.
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import shared helpers AFTER augmenting sys.path
from main import create_services, setup_logging  # type: ignore  # noqa: E402
from src.utils.config import ConfigManager  # type: ignore  # noqa: E402
from src.utils.file_utils import FileUtils  # type: ignore  # noqa: E402


DEFAULT_WORKING_DIR = (PROJECT_ROOT.parent / "working").resolve()
DEFAULT_RAW_DIR = DEFAULT_WORKING_DIR / "raw"
DEFAULT_PROCESSED_DIR = DEFAULT_WORKING_DIR / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess PDF documents into plain-text files"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Single PDF file or directory containing PDFs (default: working/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_PROCESSED_DIR,
        help="Directory to store extracted text files (default: working/processed)",
    )
    parser.add_argument(
        "--paths",
        type=Path,
        nargs="*",
        help="Explicit list of PDF files to process (overrides --input)",
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


def validate_paths(paths: Iterable[Path]) -> List[Path]:
    resolved: List[Path] = []
    for path in paths:
        candidate = path.expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(f"Path does not exist: {candidate}")
        if candidate.is_dir():
            raise ValueError(
                f"Explicit paths must be PDF files, not directories: {candidate}"
            )
        if candidate.suffix.lower() != ".pdf":
            raise ValueError(f"Only PDF files are supported: {candidate}")
        resolved.append(candidate)
    return resolved


def collect_pdf_files(directory: Path) -> List[Path]:
    """递归收集目录下的全部 PDF 文件。"""
    return sorted(p for p in directory.rglob("*.pdf") if p.is_file())


def is_token_pdf(pdf_path: Path, raw_root: Optional[Path]) -> bool:
    """
    判断文件是否位于 token-pdf 目录，用于决定是否走快速提取通道。
    """
    target = "token-pdf"
    try:
        if raw_root:
            rel = pdf_path.resolve().relative_to(raw_root.resolve())
            if target in rel.parts:
                return True
    except Exception:
        pass
    return target in [p.name for p in pdf_path.resolve().parents]


def fast_extract_text(pdf_path: Path) -> str:
    """使用 PyMuPDF 快速提取全文文本。"""
    if fitz is None:
        raise RuntimeError("PyMuPDF 未安装，无法执行快速文本提取。请安装 `pip install pymupdf`。")

    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n\n".join(pages)


def preprocess_raw_pdfs_with_paddle(
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_PROCESSED_DIR,
    config_path: Optional[Path] = PROJECT_ROOT / "config_local.yaml",
    log_level: str = "INFO",
    session_id: Optional[str] = None,
) -> None:
    """
    Preprocess every PDF in working/raw into working/processed using PaddleOCR.

    This is a convenience wrapper that mirrors the example PaddleOCR usage
    shared by the user while reusing the existing service layer so downstream
    steps (vectorization, Q&A generation) continue to work unchanged.
    """
    logger = setup_logging(log_level)
    config = ConfigManager(config_path)
    # 确保使用 PaddleOCR
    config.set("ocr.provider", "paddle")

    pdf_processor, *_ = create_services(config, logger)
    if pdf_processor is None:
        raise RuntimeError("OCR engine is not available; please install/configure PaddleOCR")

    resolved_raw = raw_dir.expanduser().resolve()
    resolved_output = output_dir.expanduser().resolve()

    if not resolved_raw.exists():
        raise FileNotFoundError(f"Input directory does not exist: {resolved_raw}")

    FileUtils.ensure_directory(resolved_output)

    logger.info("Using PaddleOCR to preprocess PDFs")
    logger.info("Input directory: %s", resolved_raw)
    logger.info("Output directory: %s", resolved_output)

    pdf_processor.process_directory(resolved_raw, resolved_output, session_id=session_id)


def main() -> None:
    args = parse_args()

    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    pdf_processor, _, _, progress_manager = create_services(config, logger)
    if pdf_processor is None:
        raise RuntimeError(
            "PDF processing is unavailable because no OCR engine is configured"
        )

    output_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(output_dir)

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths = validate_paths(args.paths)

    raw_root = args.input.expanduser().resolve()

    if explicit_paths:
        logger.info(
            "Processing %d explicitly provided PDF file(s)", len(explicit_paths)
        )
        total = len(explicit_paths)
        for index, pdf_path in enumerate(explicit_paths, start=1):
            if progress_manager.should_skip(pdf_path, "preprocess"):
                logger.info(
                    "[%d/%d] Skipping PDF (preprocess already done): %s",
                    index,
                    total,
                    pdf_path,
                )
                continue

            try:
                if is_token_pdf(pdf_path, raw_root):
                    logger.info("[%d/%d] (token-pdf 快速提取) %s", index, total, pdf_path)
                    text = fast_extract_text(pdf_path)
                    FileUtils.save_text_file(text, output_dir / f"{pdf_path.stem}.txt")
                    progress_manager.update_status(pdf_path, "preprocess", "done")
                else:
                    logger.info("[%d/%d] Processing PDF: %s", index, total, pdf_path)
                    result = pdf_processor.process_pdf(pdf_path, output_dir, None)
                    progress_manager.update_status(
                        pdf_path, "preprocess", "done" if result else "failed"
                    )
            except Exception:
                logger.exception("Failed to preprocess PDF: %s", pdf_path)
                progress_manager.update_status(pdf_path, "preprocess", "failed")
                continue
        return

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if progress_manager.should_skip(input_path, "preprocess"):
            logger.info("Skipping single PDF (preprocess already done): %s", input_path)
            return

        logger.info("Processing single PDF file: %s", input_path)
        try:
            result = pdf_processor.process_pdf(input_path, output_dir, None)
        except Exception:
            logger.exception("Failed to preprocess PDF: %s", input_path)
            progress_manager.update_status(input_path, "preprocess", "failed")
            return
        progress_manager.update_status(
            input_path, "preprocess", "done" if result else "failed"
        )
    else:
        pdf_files = collect_pdf_files(input_path)
        if not pdf_files:
            raise ValueError(f"No PDF files found under directory: {input_path}")

        logger.info("Processing all PDFs under directory: %s", input_path)
        total = len(pdf_files)
        for index, pdf_path in enumerate(pdf_files, start=1):
            if progress_manager.should_skip(pdf_path, "preprocess"):
                logger.info(
                    "[%d/%d] Skipping PDF (preprocess already done): %s",
                    index,
                    total,
                    pdf_path,
                )
                continue

            try:
                if is_token_pdf(pdf_path, raw_root):
                    logger.info("[%d/%d] (token-pdf 快速提取) %s", index, total, pdf_path)
                    text = fast_extract_text(pdf_path)
                    FileUtils.save_text_file(text, output_dir / f"{pdf_path.stem}.txt")
                    progress_manager.update_status(pdf_path, "preprocess", "done")
                else:
                    logger.info("[%d/%d] Processing PDF: %s", index, total, pdf_path)
                    result = pdf_processor.process_pdf(pdf_path, output_dir, None)
                    progress_manager.update_status(
                        pdf_path, "preprocess", "done" if result else "failed"
                    )
            except Exception:
                logger.exception("Failed to preprocess PDF: %s", pdf_path)
                progress_manager.update_status(pdf_path, "preprocess", "failed")
                continue


if __name__ == "__main__":
    main()
