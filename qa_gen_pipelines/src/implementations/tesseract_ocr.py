"""Tesseract OCR implementation."""

import pytesseract
from PIL import Image, ImageFilter
from pdf2image import convert_from_path
from pathlib import Path
from typing import List
from datetime import datetime
from loguru import logger
from tqdm import tqdm

from ..interfaces.ocr_interface import OCRInterface, OCRError
from ..models.document import Document
from ..utils.config import ConfigManager


class TesseractOCR(OCRInterface):
    """Tesseract-based OCR implementation."""
    
    def __init__(self, config: ConfigManager):
        """
        Initialize Tesseract OCR.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.lang = config.get("ocr.tesseract.lang", "chi_sim+eng")
        self.tesseract_config = config.get("ocr.tesseract.config", "--psm 6")
        self.timeout = config.get("ocr.tesseract.timeout", 30)
        self.dpi = config.get("ocr.tesseract.dpi", 300)
        self.enable_preprocess = config.get("ocr.tesseract.enable_preprocess", True)
        self.binarize_threshold = config.get("ocr.tesseract.binarize_threshold", 180)
        self.apply_median_filter = config.get("ocr.tesseract.apply_median_filter", True)
        self.min_confidence = config.get("ocr.tesseract.min_confidence", 30)  # 置信度阈值
        
        # Test tesseract installation
        try:
            pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR initialized with language: {self.lang}")
        except Exception as e:
            raise OCRError(f"Tesseract not found or not properly installed: {e}")
    
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text content
            
        Raises:
            OCRError: If extraction fails
        """
        try:
            logger.info(f"Starting OCR extraction for: {pdf_path}")
            
            # Convert PDF to images
            images = convert_from_path(str(pdf_path), dpi=self.dpi)
            logger.info(f"Converted PDF to {len(images)} images")
            
            extracted_text = []
            
            # Process each page
            for i, image in enumerate(tqdm(images, desc="Processing pages")):
                try:
                    processed_image = (
                        self._preprocess_image(image)
                        if self.enable_preprocess
                        else image
                    )

                    # 使用 image_to_data 获取置信度信息，过滤低质量识别（图片噪声）
                    data = pytesseract.image_to_data(
                        processed_image,
                        lang=self.lang,
                        config=self.tesseract_config,
                        timeout=self.timeout,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # 过滤并重建文本，只保留高置信度的识别结果
                    page_text = self._filter_text_by_confidence(data, min_conf=self.min_confidence)
                    
                    if page_text.strip():
                        extracted_text.append(f"--- Page {i + 1} ---\n{page_text.strip()}")
                        
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {i + 1}: {e}")
                    continue
            
            full_text = "\n\n".join(extracted_text)
            logger.info(f"OCR extraction completed. Total characters: {len(full_text)}")
            
            return full_text
            
        except Exception as e:
            raise OCRError(f"Failed to extract text from {pdf_path}: {e}")
    
    def process_pdf_to_document(self, pdf_path: Path) -> Document:
        """
        Process PDF and create Document object.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Document object with extracted content and metadata
            
        Raises:
            OCRError: If processing fails
        """
        try:
            # Extract text content
            content = self.extract_text_from_pdf(pdf_path)
            
            # Get file metadata
            file_size = pdf_path.stat().st_size
            created_at = datetime.fromtimestamp(pdf_path.stat().st_ctime)
            
            # Create Document object
            document = Document(
                file_path=pdf_path,
                content=content,
                file_type=pdf_path.suffix,
                file_size=file_size,
                created_at=created_at,
                processed_at=datetime.now()
            )
            
            logger.info(f"Created document object for: {pdf_path}")
            return document
            
        except Exception as e:
            raise OCRError(f"Failed to process PDF to document: {e}")
    
    def process_batch(self, pdf_paths: List[Path], output_dir: Path) -> List[Document]:
        """
        Process multiple PDF files in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            output_dir: Directory to save extracted text files
            
        Returns:
            List of processed Document objects
            
        Raises:
            OCRError: If batch processing fails
        """
        try:
            documents = []
            output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"Starting batch OCR processing for {len(pdf_paths)} files")
            
            for pdf_path in tqdm(pdf_paths, desc="Processing PDFs"):
                try:
                    # Process PDF to document
                    document = self.process_pdf_to_document(pdf_path)
                    documents.append(document)
                    
                    # Save extracted text to file
                    text_filename = pdf_path.stem + ".txt"
                    text_path = output_dir / text_filename
                    
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(document.content)
                    
                    logger.info(f"Saved extracted text to: {text_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to process {pdf_path}: {e}")
                    continue
            
            logger.info(f"Batch processing completed. Processed {len(documents)} documents")
            return documents
            
        except Exception as e:
            raise OCRError(f"Batch processing failed: {e}")

    def _filter_text_by_confidence(self, data: dict, min_conf: float = 30) -> str:
        """
        过滤低置信度的识别结果，减少图片产生的噪声。
        
        Args:
            data: pytesseract.image_to_data 返回的字典
            min_conf: 最低置信度阈值（0-100），低于此值的文本将被忽略
        
        Returns:
            过滤后的文本
        """
        import re
        
        lines = {}  # {block_num: {line_num: [words]}}
        
        for i in range(len(data['text'])):
            conf = float(data['conf'][i])
            text = data['text'][i].strip()
            
            # 跳过空文本或低置信度文本
            if not text or conf < min_conf:
                continue
            
            # 跳过纯符号/噪声（只包含特殊字符、单个字母等）
            if len(text) <= 2 and not re.search(r'[\u4e00-\u9fff]', text):  # 不是中文且长度<=2
                continue
            
            # 跳过明显的噪声模式
            if re.match(r'^[_\-=+|/\\<>~`\'\",.;:!?@#$%^&*()[\]{}]+$', text):
                continue
            
            block_num = data['block_num'][i]
            line_num = data['line_num'][i]
            
            if block_num not in lines:
                lines[block_num] = {}
            if line_num not in lines[block_num]:
                lines[block_num][line_num] = []
            
            lines[block_num][line_num].append(text)
        
        # 重建文本，保持原有的行结构
        result_lines = []
        for block_num in sorted(lines.keys()):
            for line_num in sorted(lines[block_num].keys()):
                line_text = ' '.join(lines[block_num][line_num])
                if line_text.strip():
                    result_lines.append(line_text)
        
        return '\n'.join(result_lines)
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image to improve OCR accuracy.
        Converts to grayscale, optional denoising, and binarizes via threshold.
        """
        gray = image.convert("L")
        if self.apply_median_filter:
            gray = gray.filter(ImageFilter.MedianFilter(size=3))

        # Binarize image using simple threshold
        threshold = max(0, min(255, int(self.binarize_threshold)))
        bw = gray.point(lambda x: 255 if x > threshold else 0, "1")
        return bw
    
    def is_supported_format(self, file_path: Path) -> bool:
        """
        Check if file format is supported.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if format is supported, False otherwise
        """
        supported_extensions = ['.pdf']
        return file_path.suffix.lower() in supported_extensions 