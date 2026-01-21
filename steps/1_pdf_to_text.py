#!/usr/bin/env python3
"""
步骤1: PDF转文本
读取PDF -> OCR -> 存入 working/processed

支持 --domain 参数指定行业/领域子目录
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from config import load_settings
from src.ocr_engine import OCREngine


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def list_available_domains(raw_dir: Path) -> list:
    """列出可用的domain目录"""
    if not raw_dir.exists():
        return []
    domains = [d.name for d in raw_dir.iterdir() if d.is_dir()]
    return sorted(domains)


def main():
    parser = argparse.ArgumentParser(description="步骤1: PDF转文本 (OCR)")
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=PROJECT_ROOT / "config" / "config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--domain", "-d",
        type=str,
        help="指定行业/领域子目录 (如: Robot, Numerical-Control-System)"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        help="输入PDF文件或目录 (默认: working/raw/[domain])"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="输出目录 (默认: working/processed/[domain])"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="递归处理子目录"
    )
    parser.add_argument(
        "--list-domains",
        action="store_true",
        help="列出可用的domain目录"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别"
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    # 加载配置
    settings = load_settings(args.config)
    paths = settings.get_working_paths(PROJECT_ROOT)

    # 列出可用domains
    if args.list_domains:
        domains = list_available_domains(paths["raw_dir"])
        if domains:
            print("可用的 domain 目录:")
            for d in domains:
                print(f"  - {d}")
        else:
            print(f"未找到 domain 目录 (在 {paths['raw_dir']} 下)")
        return

    # 确定输入输出路径
    if args.input:
        input_path = args.input
    elif args.domain:
        input_path = paths["raw_dir"] / args.domain
    else:
        input_path = paths["raw_dir"]

    if args.output:
        output_path = args.output
    elif args.domain:
        output_path = paths["processed_dir"] / args.domain
    else:
        output_path = paths["processed_dir"]

    # 确保目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("步骤1: PDF转文本 (OCR)")
    logger.info("=" * 60)
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"输出路径: {output_path}")

    # 检查输入路径
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        if args.domain:
            available = list_available_domains(paths["raw_dir"])
            if available:
                logger.info(f"可用的 domain: {', '.join(available)}")
        sys.exit(1)

    # 初始化OCR引擎
    ocr = OCREngine(
        lang=settings.ocr.lang,
        use_angle_cls=settings.ocr.use_angle_cls,
        dpi=settings.ocr.dpi,
    )

    if not ocr.is_available():
        logger.error("OCR引擎不可用，请安装 PaddleOCR")
        sys.exit(1)

    # 处理PDF
    if input_path.is_file():
        if input_path.suffix.lower() != ".pdf":
            logger.error(f"不支持的文件格式: {input_path.suffix}")
            sys.exit(1)
        
        logger.info(f"处理单个PDF文件: {input_path.name}")
        result = ocr.process_pdf(input_path, output_path)
        
        if result:
            logger.info(f"✅ 处理完成: {input_path.name}")
            logger.info(f"   文本长度: {len(result['content'])} 字符")
        else:
            logger.error(f"❌ 处理失败: {input_path.name}")
            sys.exit(1)
    else:
        logger.info(f"批量处理PDF目录: {input_path}")
        results = ocr.process_directory(input_path, output_path, recursive=args.recursive)
        
        logger.info("=" * 60)
        logger.info("处理完成")
        logger.info(f"成功处理: {len(results)} 个PDF文件")
        logger.info(f"输出目录: {output_path}")
        logger.info("=" * 60)

    logger.info("步骤1完成！")


if __name__ == "__main__":
    main()
