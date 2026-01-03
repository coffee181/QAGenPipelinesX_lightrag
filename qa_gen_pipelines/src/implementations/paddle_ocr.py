"""PPStructureV3-based OCR implementation."""

from __future__ import annotations

from datetime import datetime
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from loguru import logger
from paddleocr import PPStructureV3
from tqdm import tqdm

from ..interfaces.ocr_interface import OCRError, OCRInterface
from ..models.document import Document
from ..utils.config import ConfigManager
from .simple_markdown_processor import SimpleMarkdownProcessor


class PaddleOCREngine(OCRInterface):
    """
    OCR implementation using PaddleOCR's PPStructureV3 for structured PDF parsing.
    """

    def __init__(self, config: ConfigManager):
        self.config = config
        paddle_cfg = config.get("ocr.paddle", {}) or {}
        self.lang = paddle_cfg.get("lang", "ch")
        self.use_angle_cls = paddle_cfg.get("use_angle_cls", True)

        try:
            self.pipeline = PPStructureV3(lang=self.lang)
            logger.info("PPStructureV3 initialized (lang=%s)", self.lang)
        except Exception as e:
            raise OCRError(f"Failed to initialize PPStructureV3: {e}")

        # Markdown -> plain text converter to feed downstream chunker/Q&A
        self.markdown_processor = SimpleMarkdownProcessor()

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract plain text from a PDF using PPStructureV3.
        """
        markdown_text, plain_text, _, _ = self._process_pdf(pdf_path, output_dir=None)
        return plain_text or markdown_text

    def process_pdf_to_document(self, pdf_path: Path, output_dir: Optional[Path] = None) -> Document:
        """
        Process a PDF into a Document, saving Markdown/images when output_dir is provided.
        """
        try:
            markdown_text, plain_text, table_markdown, image_texts = self._process_pdf(
                pdf_path, output_dir=output_dir
            )

            # 去重主文本，附加表格和图片 OCR，避免重复向量化
            content_parts: List[str] = []
            if plain_text:
                content_parts.append(plain_text)
            else:
                content_parts.append(self.markdown_processor.markdown_to_plain_text(markdown_text))

            if table_markdown:
                content_parts.append("## 表格提取\n" + "\n\n".join(table_markdown))

            if image_texts:
                content_parts.append("## 图片OCR\n" + "\n\n".join(image_texts))

            content = "\n\n".join([c for c in content_parts if c.strip()])

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
            raise OCRError(f"Failed to process PDF with PPStructureV3: {e}")

    def process_batch(self, pdf_paths: List[Path], output_dir: Path) -> List[Document]:
        """
        Batch process PDFs; saves both .txt (plain) and .md (structured) plus images.
        """
        try:
            documents: List[Document] = []
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info("Starting PPStructureV3 batch processing for %d files", len(pdf_paths))

            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs with PPStructureV3"):
                try:
                    document = self.process_pdf_to_document(pdf_path, output_dir=output_dir)
                    documents.append(document)

                    text_filename = pdf_path.stem + ".txt"
                    text_path = output_dir / text_filename
                    text_path.write_text(document.content, encoding="utf-8")

                    logger.info("Saved PPStructureV3 extracted text to: %s", text_path)
                except Exception as e:
                    logger.error("Failed to process %s with PPStructureV3: %s", pdf_path, e)
                    continue

            logger.info("PPStructureV3 batch processing completed. Processed %d documents", len(documents))
            return documents

        except Exception as e:
            raise OCRError(f"PPStructureV3 batch processing failed: {e}")

    def is_supported_format(self, file_path: Path) -> bool:
        """Check if file format is supported."""
        supported_extensions = [".pdf"]
        return file_path.suffix.lower() in supported_extensions

    def _process_pdf(self, pdf_path: Path, output_dir: Optional[Path] = None):
        """
        Run PPStructureV3 on a PDF and optionally save markdown and images.
        """
        logger.info("Starting PPStructureV3 extraction for: %s", pdf_path)

        output = self.pipeline.predict(input=str(pdf_path))
        if not output:
            raise OCRError(f"PPStructureV3 returned empty result for {pdf_path}")

        markdown_pages: List[Any] = []
        markdown_images: List[Dict[str, Any]] = []

        for res in output:
            if hasattr(res, "markdown"):
                md_info = res.markdown
                markdown_pages.append(md_info)

                if isinstance(md_info, dict):
                    markdown_images.append(md_info.get("markdown_images", {}) or {})
                elif hasattr(res, "markdown_images"):
                    markdown_images.append(getattr(res, "markdown_images") or {})
                else:
                    markdown_images.append({})

        table_markdown = self._extract_table_markdown(output)
        markdown_text = self._concat_markdown_pages(markdown_pages)
        if table_markdown:
            markdown_text = f"{markdown_text}\n\n" + "\n\n".join(table_markdown)

        # 先移除 img 标签，避免在纯文本中留下占位符
        cleaned_markdown = re.sub(r"<img[^>]*>", " ", markdown_text, flags=re.IGNORECASE | re.DOTALL)
        plain_text = self.markdown_processor.markdown_to_plain_text(cleaned_markdown)

        image_texts: List[str] = []
        if output_dir is not None:
            image_texts = self._save_markdown_and_images(
                pdf_path=pdf_path,
                base_output_dir=output_dir,
                markdown_text=markdown_text,
                markdown_images=markdown_images,
            )

        logger.info(
            "PPStructureV3 extraction completed. Markdown length: %d, Plain length: %d",
            len(markdown_text),
            len(plain_text),
        )
        return markdown_text, plain_text, table_markdown, image_texts

    def _concat_markdown_pages(self, markdown_pages: List[Any]) -> str:
        """Concatenate page-level markdown outputs safely."""
        if hasattr(self.pipeline, "concatenate_markdown_pages"):
            try:
                return self.pipeline.concatenate_markdown_pages(markdown_pages)
            except Exception as e:
                logger.warning("Failed to use concatenate_markdown_pages: %s", e)

        normalized = [self._markdown_to_string(md) for md in markdown_pages]
        return "\n\n".join([m for m in normalized if m])

    @staticmethod
    def _markdown_to_string(md_info: Any) -> str:
        """Normalize markdown output to a string."""
        if md_info is None:
            return ""
        if isinstance(md_info, str):
            return md_info
        if isinstance(md_info, dict):
            # Common keys used by PPStructureV3 markdown output
            if "markdown" in md_info and isinstance(md_info["markdown"], str):
                return md_info["markdown"]
            if "md" in md_info and isinstance(md_info["md"], str):
                return md_info["md"]
            if "text" in md_info and isinstance(md_info["text"], str):
                return md_info["text"]
        return str(md_info)

    def _extract_table_markdown(self, output: List[Any]) -> List[str]:
        """
        Extract table results from PPStructureV3 output and convert to markdown.
        """
        tables: List[str] = []

        for res in output:
            candidate = getattr(res, "res", None)
            if not candidate:
                continue

            # Case 1: list of dict results
            if isinstance(candidate, list):
                for item in candidate:
                    html = self._get_table_html(item)
                    if html:
                        self._append_table_from_html(html, tables)

            # Case 2: dict result
            if isinstance(candidate, dict):
                html = self._get_table_html(candidate)
                if html:
                    self._append_table_from_html(html, tables)

        return tables

    @staticmethod
    def _get_table_html(item: Dict[str, Any]) -> Optional[str]:
        if not isinstance(item, dict):
            return None
        return (
            item.get("res", {}).get("html")
            or item.get("html")
            or item.get("cell_html")
        )

    @staticmethod
    def _append_table_from_html(html: str, tables: List[str]) -> None:
        if not html:
            return
        try:
            dfs = pd.read_html(html)
            if dfs:
                tables.append(dfs[0].to_markdown(index=False))
        except Exception as e:
            logger.debug("Failed to parse table html: %s", e)

    def _save_markdown_and_images(
        self,
        pdf_path: Path,
        base_output_dir: Path,
        markdown_text: str,
        markdown_images: List[Dict[str, Any]],
    ) -> List[str]:
        """
        Save markdown (.md) and extracted images to disk, mirroring the reference sample.
        """
        base_output_dir.mkdir(parents=True, exist_ok=True)

        # Save markdown file
        md_path = base_output_dir / f"{pdf_path.stem}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        logger.info("Markdown saved: %s", md_path)

        image_texts: List[str] = []

        # Save images
        for item in markdown_images:
            if not item:
                continue
            for img_rel_path, image in item.items():
                file_path = base_output_dir / img_rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    image.save(file_path)
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        image_texts.append(f"[图片OCR] {img_rel_path}: {ocr_text}")
                except Exception as e:
                    logger.warning("Failed to save image %s: %s", file_path, e)

        return image_texts

    def _ocr_image(self, image) -> str:
        """Run OCR on a PIL image using Tesseract and return concatenated text."""
        try:
            import pytesseract

            text = pytesseract.image_to_string(image, lang=self.lang)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Image OCR (tesseract) failed: %s", exc)
            return ""

