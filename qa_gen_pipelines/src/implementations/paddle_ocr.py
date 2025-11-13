"""PaddleOCR implementation."""

from __future__ import annotations

import numpy as np
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
from pathlib import Path
from typing import List
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from ..interfaces.ocr_interface import OCRInterface, OCRError
from ..models.document import Document
from ..utils.config import ConfigManager


class PaddleOCREngine(OCRInterface):
    """PaddleOCR-based OCR implementation."""

    def __init__(self, config: ConfigManager):
        """
        Initialize PaddleOCR.

        Args:
            config: Configuration object.
        """
        self.config = config
        paddle_cfg = config.get("ocr.paddle", {}) or {}

        self.lang = paddle_cfg.get("lang", "ch")
        self.use_angle_cls = paddle_cfg.get("use_angle_cls", True)
        self.dpi = paddle_cfg.get("dpi", 300)

        # Initialize PaddleOCR engine
        try:
            self.ocr = PaddleOCR(lang=self.lang, use_angle_cls=self.use_angle_cls)
            logger.info(
                "PaddleOCR initialized (lang=%s, angle_cls=%s, dpi=%s)",
                self.lang,
                self.use_angle_cls,
                self.dpi,
            )
        except Exception as e:
            raise OCRError(f"Failed to initialize PaddleOCR: {e}")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF file using PaddleOCR.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Extracted text content.
        """
        try:
            logger.info("Starting PaddleOCR extraction for: %s", pdf_path)

            images = convert_from_path(str(pdf_path), dpi=self.dpi)
            logger.info("Converted PDF to %d images", len(images))

            extracted_text: List[str] = []
            for i, image in enumerate(tqdm(images, desc="PaddleOCR pages")):
                page_text = self._extract_text_from_image(image)
                if page_text:
                    extracted_text.append(f"--- Page {i + 1} ---\n{page_text}")

            full_text = "\n\n".join(extracted_text)
            logger.info(
                "PaddleOCR extraction completed. Total characters: %d", len(full_text)
            )
            return full_text

        except Exception as e:
            raise OCRError(f"Failed to extract text from {pdf_path} using PaddleOCR: {e}")

    def _extract_text_from_image(self, image) -> str:
        """Run PaddleOCR on a PIL image."""
        np_img = np.array(image)
        result = self.ocr.ocr(np_img)
        if not result:
            return ""

        lines: List[str] = []
        for detection in result:
            if not detection:
                continue
            for line in detection:
                text = line[1][0] if isinstance(line, (list, tuple)) else ""
                if text and text.strip():
                    lines.append(text.strip())

        return "\n".join(lines)

    def process_pdf_to_document(self, pdf_path: Path) -> Document:
        """
        Process PDF and create Document object.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Document with extracted content and metadata.
        """
        try:
            content = self.extract_text_from_pdf(pdf_path)

            file_size = pdf_path.stat().st_size
            created_at = datetime.fromtimestamp(pdf_path.stat().st_ctime)

            document = Document(
                file_path=pdf_path,
                content=content,
                file_type=pdf_path.suffix,
                file_size=file_size,
                created_at=created_at,
                processed_at=datetime.now(),
            )

            logger.info("Created document object for: %s", pdf_path)
            return document

        except Exception as e:
            raise OCRError(f"Failed to process PDF to document with PaddleOCR: {e}")

    def process_batch(self, pdf_paths: List[Path], output_dir: Path) -> List[Document]:
        """
        Process multiple PDF files in batch.

        Args:
            pdf_paths: List of PDF file paths.
            output_dir: Directory to save extracted text files.

        Returns:
            List of processed Document objects.
        """
        try:
            documents: List[Document] = []
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Starting PaddleOCR batch processing for %d files", len(pdf_paths))

            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs with PaddleOCR"):
                try:
                    document = self.process_pdf_to_document(pdf_path)
                    documents.append(document)

                    text_filename = pdf_path.stem + ".txt"
                    text_path = output_dir / text_filename
                    with open(text_path, "w", encoding="utf-8") as f:
                        f.write(document.content)

                    logger.info("Saved PaddleOCR extracted text to: %s", text_path)

                except Exception as e:
                    logger.error("Failed to process %s with PaddleOCR: %s", pdf_path, e)
                    continue

            logger.info("PaddleOCR batch processing completed. Processed %d documents", len(documents))
            return documents

        except Exception as e:
            raise OCRError(f"PaddleOCR batch processing failed: {e}")

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        supported_extensions = [".pdf"]
        return file_path.suffix.lower() in supported_extensions

