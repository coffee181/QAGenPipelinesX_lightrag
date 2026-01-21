"""OCR引擎模块 - 封装PaddleOCR (PPStructureV3)"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger

# PaddleOCR 导入
try:
    from paddleocr import PPStructureV3
    import pandas as pd
    PADDLE_AVAILABLE = True
except ImportError:
    PPStructureV3 = None
    pd = None
    PADDLE_AVAILABLE = False
    logger.warning("PaddleOCR 未安装，OCR功能不可用")


class OCREngine:
    """
    OCR引擎 - 使用PaddleOCR的PPStructureV3进行PDF结构化解析
    """

    def __init__(
        self, 
        lang: str = "ch", 
        use_angle_cls: bool = True, 
        dpi: int = 300,
    ):
        """
        初始化OCR引擎
        
        Args:
            lang: 语言代码 (ch=中文)
            use_angle_cls: 是否使用角度分类
            dpi: PDF渲染DPI
        """
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.dpi = dpi
        self.pipeline = None
        
        if PADDLE_AVAILABLE:
            self._init_pipeline()
        else:
            logger.error("PaddleOCR 不可用，请安装: pip install paddleocr paddlex")

    def _init_pipeline(self):
        """初始化 PPStructureV3 管道"""
        logger.info(f"初始化 PPStructureV3 (lang={self.lang})...")
        self.pipeline = PPStructureV3(lang=self.lang)
        logger.info(f"✅ PPStructureV3 初始化完成")

    def is_available(self) -> bool:
        """检查OCR引擎是否可用"""
        return self.pipeline is not None

    def extract_text(self, pdf_path: Path) -> str:
        """
        从PDF提取纯文本
        
        Args:
            pdf_path: PDF文件路径
            
        Returns:
            提取的纯文本
        """
        markdown_text, plain_text, _, _ = self._process_pdf(pdf_path)
        return plain_text or markdown_text

    def process_pdf(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        处理PDF文件，提取文本并可选保存结果
        
        Args:
            pdf_path: PDF文件路径
            output_dir: 输出目录（可选）
            
        Returns:
            包含处理结果的字典
        """
        if not self.is_available():
            raise RuntimeError("PaddleOCR 不可用")

        logger.info(f"开始处理PDF: {pdf_path.name}")
        
        markdown_text, plain_text, table_markdown, image_texts = self._process_pdf(
            pdf_path, output_dir
        )

        # 合并所有内容
        content_parts: List[str] = []
        if plain_text:
            content_parts.append(plain_text)
        else:
            content_parts.append(self._markdown_to_plain(markdown_text))

        if table_markdown:
            content_parts.append("## 表格提取\n" + "\n\n".join(table_markdown))

        if image_texts:
            content_parts.append("## 图片OCR\n" + "\n\n".join(image_texts))

        content = "\n\n".join([c for c in content_parts if c.strip()])

        # 保存文本文件
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            txt_path = output_dir / f"{pdf_path.stem}.txt"
            txt_path.write_text(content, encoding="utf-8")
            logger.info(f"已保存文本: {txt_path}")

        logger.info(f"PDF处理完成: {pdf_path.name} ({len(content)} 字符)")
        
        return {
            "file_path": pdf_path,
            "content": content,
            "markdown": markdown_text,
            "tables": table_markdown,
            "images": image_texts,
        }

    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        recursive: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        批量处理目录中的PDF文件
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            recursive: 是否递归处理子目录
            
        Returns:
            处理结果列表
        """
        if not self.is_available():
            raise RuntimeError("PaddleOCR 不可用")

        # 查找所有PDF文件
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = sorted(input_dir.glob(pattern))

        if not pdf_files:
            logger.warning(f"目录中未找到PDF文件: {input_dir}")
            return []

        logger.info(f"发现 {len(pdf_files)} 个PDF文件")
        
        results = []
        for i, pdf_path in enumerate(pdf_files, 1):
            logger.info(f"[{i}/{len(pdf_files)}] 处理: {pdf_path.name}")
            
            # 保持子目录结构
            rel_path = pdf_path.relative_to(input_dir)
            sub_output_dir = output_dir / rel_path.parent
            
            result = self.process_pdf(pdf_path, sub_output_dir)
            if result:
                results.append(result)

        logger.info(f"批量处理完成: {len(results)}/{len(pdf_files)} 个文件")
        return results

    def _process_pdf(
        self, pdf_path: Path, output_dir: Optional[Path] = None
    ) -> tuple:
        """使用 PPStructureV3 处理 PDF"""
        logger.info(f"PPStructureV3 开始提取: {pdf_path}")

        output = self.pipeline.predict(input=str(pdf_path))
        if not output:
            raise RuntimeError(f"PPStructureV3 返回空结果: {pdf_path}")

        # 收集Markdown页面和图片
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

        # 提取表格
        table_markdown = self._extract_tables(output)
        
        # 合并Markdown
        markdown_text = self._concat_markdown_pages(markdown_pages)
        if table_markdown:
            markdown_text = f"{markdown_text}\n\n" + "\n\n".join(table_markdown)

        # 转换为纯文本
        cleaned_markdown = re.sub(
            r"<img[^>]*>", " ", markdown_text, flags=re.IGNORECASE | re.DOTALL
        )
        plain_text = self._markdown_to_plain(cleaned_markdown)

        # 保存Markdown和图片
        image_texts: List[str] = []
        if output_dir is not None:
            image_texts = self._save_outputs(
                pdf_path, output_dir, markdown_text, markdown_images
            )

        logger.info(
            f"提取完成: Markdown {len(markdown_text)} 字符, 纯文本 {len(plain_text)} 字符"
        )
        return markdown_text, plain_text, table_markdown, image_texts

    def _concat_markdown_pages(self, markdown_pages: List[Any]) -> str:
        """合并多页Markdown"""
        if hasattr(self.pipeline, "concatenate_markdown_pages"):
            try:
                return self.pipeline.concatenate_markdown_pages(markdown_pages)
            except Exception as e:
                logger.warning(f"使用 concatenate_markdown_pages 失败: {e}")

        normalized = [self._md_to_string(md) for md in markdown_pages]
        return "\n\n".join([m for m in normalized if m])

    @staticmethod
    def _md_to_string(md_info: Any) -> str:
        """将Markdown输出规范化为字符串"""
        if md_info is None:
            return ""
        if isinstance(md_info, str):
            return md_info
        if isinstance(md_info, dict):
            for key in ("markdown", "md", "text"):
                if key in md_info and isinstance(md_info[key], str):
                    return md_info[key]
        return str(md_info)

    def _extract_tables(self, output: List[Any]) -> List[str]:
        """从输出中提取表格并转换为Markdown"""
        tables: List[str] = []

        for res in output:
            candidate = getattr(res, "res", None)
            if not candidate:
                continue

            if isinstance(candidate, list):
                for item in candidate:
                    html = self._get_table_html(item)
                    if html:
                        self._append_table(html, tables)

            if isinstance(candidate, dict):
                html = self._get_table_html(candidate)
                if html:
                    self._append_table(html, tables)

        return tables

    @staticmethod
    def _get_table_html(item: Dict[str, Any]) -> Optional[str]:
        """获取表格HTML"""
        if not isinstance(item, dict):
            return None
        return (
            item.get("res", {}).get("html")
            or item.get("html")
            or item.get("cell_html")
        )

    @staticmethod
    def _append_table(html: str, tables: List[str]) -> None:
        """将HTML表格转换为Markdown并添加到列表"""
        if not html or pd is None:
            return
        try:
            dfs = pd.read_html(html)
            if dfs:
                tables.append(dfs[0].to_markdown(index=False))
        except Exception as e:
            logger.debug(f"解析表格HTML失败: {e}")

    def _save_outputs(
        self,
        pdf_path: Path,
        output_dir: Path,
        markdown_text: str,
        markdown_images: List[Dict[str, Any]],
    ) -> List[str]:
        """保存Markdown和图片"""
        output_dir.mkdir(parents=True, exist_ok=True)

        # 保存Markdown
        md_path = output_dir / f"{pdf_path.stem}.md"
        md_path.write_text(markdown_text, encoding="utf-8")
        logger.info(f"Markdown已保存: {md_path}")

        image_texts: List[str] = []

        # 保存图片
        for item in markdown_images:
            if not item:
                continue
            for img_rel_path, image in item.items():
                file_path = output_dir / img_rel_path
                file_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    image.save(file_path)
                    ocr_text = self._ocr_image(image)
                    if ocr_text:
                        image_texts.append(f"[图片OCR] {img_rel_path}: {ocr_text}")
                except Exception as e:
                    logger.warning(f"保存图片失败 {file_path}: {e}")

        return image_texts

    def _ocr_image(self, image) -> str:
        """对图片进行OCR"""
        try:
            import pytesseract
            text = pytesseract.image_to_string(image, lang=self.lang)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"图片OCR失败: {e}")
            return ""

    def _markdown_to_plain(self, markdown_text: str) -> str:
        """将Markdown转换为纯文本"""
        if not markdown_text:
            return ""

        text = markdown_text

        # 处理HTML表格
        text = self._convert_html_tables(text)
        
        # 移除img标签
        text = re.sub(r"<img[^>]*>", " ", text, flags=re.IGNORECASE | re.DOTALL)
        
        # 移除HTML标签
        text = re.sub(r"<[^>]+>", " ", text)

        # 移除Markdown格式
        text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)  # 标题
        text = re.sub(r'\*\*\*(.*?)\*\*\*', r'\1', text)  # 粗斜体
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # 粗体
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # 斜体
        text = re.sub(r'`(.*?)`', r'\1', text)  # 行内代码
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)  # 代码块
        text = re.sub(r'\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # 链接
        text = re.sub(r'!\[([^\]]*)\]\([^\)]*\)', r'\1', text)  # 图片
        text = re.sub(r'^\s*[-*+]\s*', '', text, flags=re.MULTILINE)  # 列表
        text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)  # 编号列表
        text = re.sub(r'^>\s*', '', text, flags=re.MULTILINE)  # 引用
        text = re.sub(r'\|', ' ', text)  # 表格分隔符
        text = re.sub(r'^[-\s|:]+$', '', text, flags=re.MULTILINE)  # 表格分隔行

        # 清理空白
        text = re.sub(r' +', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        text = text.strip()

        return text

    def _convert_html_tables(self, text: str) -> str:
        """将HTML表格转换为Markdown"""
        if "<table" not in text.lower() or pd is None:
            return text

        def replace_table(match: re.Match) -> str:
            html = match.group(0)
            try:
                dfs = pd.read_html(html)
                if dfs:
                    return "\n" + dfs[0].to_markdown(index=False) + "\n"
            except Exception as e:
                logger.debug(f"解析HTML表格失败: {e}")
            return re.sub(r"<[^>]+>", " ", html)

        pattern = re.compile(r"<table.*?>.*?</table>", flags=re.IGNORECASE | re.DOTALL)
        return pattern.sub(replace_table, text)

