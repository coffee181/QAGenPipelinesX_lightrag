#!/usr/bin/env python3
"""
步骤2: 构建知识图谱
读取processed -> LightRAG 建图/向量化 -> 存入 working/lightrag_db

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
from src.rag_core import RAGCore


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def list_available_domains(processed_dir: Path) -> list:
    """列出可用的domain目录"""
    if not processed_dir.exists():
        return []
    domains = [d.name for d in processed_dir.iterdir() if d.is_dir()]
    return sorted(domains)


def main():
    parser = argparse.ArgumentParser(description="步骤2: 构建知识图谱 (LightRAG)")
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
        help="输入文本目录 (默认: working/processed/[domain])"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="LightRAG存储目录 (默认: working/lightrag_db/[domain])"
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
        domains = list_available_domains(paths["processed_dir"])
        if domains:
            print("可用的 domain 目录 (已有OCR文本):")
            for d in domains:
                print(f"  - {d}")
        else:
            print(f"未找到 domain 目录 (在 {paths['processed_dir']} 下)")
            print("请先运行步骤1 (1_pdf_to_text.py)")
        return

    # 确定输入输出路径
    if args.input:
        input_path = args.input
    elif args.domain:
        input_path = paths["processed_dir"] / args.domain
    else:
        input_path = paths["processed_dir"]

    if args.output:
        output_path = args.output
    elif args.domain:
        output_path = paths["lightrag_db_dir"] / args.domain
    else:
        output_path = paths["lightrag_db_dir"]

    # 确保目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("步骤2: 构建知识图谱 (LightRAG)")
    logger.info("=" * 60)
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"输入路径: {input_path}")
    logger.info(f"LightRAG存储: {output_path}")

    # 检查输入路径
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        if args.domain:
            available = list_available_domains(paths["processed_dir"])
            if available:
                logger.info(f"可用的 domain: {', '.join(available)}")
            else:
                logger.info("请先运行步骤1 (1_pdf_to_text.py) 处理PDF文件")
        sys.exit(1)

    # 查找文本文件
    pattern = "**/*.txt" if args.recursive else "*.txt"
    text_files = sorted(input_path.glob(pattern))

    if not text_files:
        logger.error(f"未找到文本文件: {input_path}")
        sys.exit(1)

    logger.info(f"发现 {len(text_files)} 个文本文件")

    # 初始化RAGCore
    rag = RAGCore(
        working_dir=output_path,
        embedding_config={
            "provider": settings.lightrag.embedding.provider,
            "model": settings.lightrag.embedding.model,
            "dim": settings.lightrag.embedding.dim,
            "base_url": settings.lightrag.embedding.base_url,
            "timeout": settings.lightrag.embedding.timeout,
            "max_retries": settings.lightrag.embedding.max_retries,
        },
        llm_config={
            "base_url": settings.lightrag.llm.base_url,
            "model": settings.lightrag.llm.model,
            "temperature": settings.lightrag.llm.temperature,
            "max_tokens": settings.lightrag.llm.max_tokens,
            "timeout": settings.lightrag.llm.timeout,
            "max_retries": settings.lightrag.llm.max_retries,
        },
        query_config={
            "top_k": settings.lightrag.query.top_k,
            "chunk_top_k": settings.lightrag.query.chunk_top_k,
            "max_entity_tokens": settings.lightrag.query.max_entity_tokens,
            "max_relation_tokens": settings.lightrag.query.max_relation_tokens,
            "max_total_tokens": settings.lightrag.query.max_total_tokens,
            "cosine_threshold": settings.lightrag.query.cosine_threshold,
            "related_chunk_number": settings.lightrag.query.related_chunk_number,
        },
    )

    # 批量插入文档
    logger.info("开始构建知识图谱...")
    rag.insert_directory(input_path, recursive=args.recursive)

    # 显示统计信息
    stats = rag.get_stats()
    logger.info("=" * 60)
    logger.info("构建完成")
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"工作目录: {stats.get('working_directory')}")
    logger.info(f"文件数量: {stats.get('file_count', 0)}")
    logger.info(f"存储大小: {stats.get('directory_size_mb', 0):.2f} MB")
    logger.info("=" * 60)

    logger.info("步骤2完成！")


if __name__ == "__main__":
    main()
