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
from typing import Iterable, List


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
        default=PROJECT_ROOT / "config.yaml",
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


def main() -> None:
    args = parse_args()

    logger = setup_logging(args.log_level)
    config = ConfigManager(args.config)

    pdf_processor, *_ = create_services(config, logger)
    if pdf_processor is None:
        raise RuntimeError(
            "PDF processing is unavailable because no OCR engine is configured"
        )

    output_dir = args.output.expanduser().resolve()
    FileUtils.ensure_directory(output_dir)

    explicit_paths: List[Path] = []
    if args.paths:
        explicit_paths = validate_paths(args.paths)

    if explicit_paths:
        logger.info(
            "Processing %d explicitly provided PDF file(s)", len(explicit_paths)
        )
        for pdf_path in explicit_paths:
            logger.info("Processing PDF: %s", pdf_path)
            pdf_processor.process_pdf(pdf_path, output_dir, args.session_id)
        return

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        logger.info("Processing single PDF file: %s", input_path)
        pdf_processor.process_pdf(input_path, output_dir, args.session_id)
    else:
        logger.info("Processing all PDFs under directory: %s", input_path)
        pdf_processor.process_directory(input_path, output_dir, args.session_id)


if __name__ == "__main__":
    main()
