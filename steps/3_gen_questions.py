#!/usr/bin/env python3
"""
步骤3: 生成问题
加载 lightrag_db -> 遍历Chunk -> 生成问题 -> 存入 working/output

支持 --domain 参数指定行业/领域子目录
"""

import sys
import json
import hashlib
import re
import argparse
from pathlib import Path
from typing import List, Set

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger
from tqdm import tqdm
from config import load_settings
from src.rag_core import RAGCore
from src.llm_client import LLMClient, TextChunker, Question, QuestionSet


def setup_logging(level: str = "INFO"):
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        colorize=True,
    )


def list_available_domains(lightrag_db_dir: Path) -> list:
    """列出可用的domain目录（已有知识图谱）"""
    if not lightrag_db_dir.exists():
        return []
    domains = [d.name for d in lightrag_db_dir.iterdir() if d.is_dir()]
    return sorted(domains)


def deduplicate_questions(
    questions: List[Question],
    similarity_threshold: float = 0.85
) -> List[Question]:
    """问题去重"""
    if not questions:
        return questions

    def normalize(text: str) -> str:
        text = re.sub(r'^问题\d+[:：]\s*', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        text = text.rstrip('?？.。！!,，')
        return text

    def get_ngrams(text: str, n: int = 2) -> set:
        return set(text[i:i+n] for i in range(len(text)-n+1))

    def similarity(text1: str, text2: str) -> float:
        ngrams1 = get_ngrams(text1)
        ngrams2 = get_ngrams(text2)
        if not ngrams1 or not ngrams2:
            return 0.0
        intersection = ngrams1 & ngrams2
        union = ngrams1 | ngrams2
        return len(intersection) / len(union) if union else 0.0

    unique_questions = []
    seen_hashes: Set[str] = set()

    for question in questions:
        normalized = normalize(question.content)
        question_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()

        if question_hash in seen_hashes:
            continue

        is_duplicate = False
        for unique_q in unique_questions:
            unique_normalized = normalize(unique_q.content)
            if similarity(normalized, unique_normalized) >= similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique_questions.append(question)
            seen_hashes.add(question_hash)

    return unique_questions


def save_questions_jsonl(question_set: QuestionSet, output_file: Path):
    """保存问题到JSONL文件"""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for q_data in question_set.to_jsonl():
            f.write(json.dumps(q_data, ensure_ascii=False))
            f.write("\n")

    logger.info(f"已保存 {question_set.total_questions} 个问题到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="步骤3: 生成问题")
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
        "--lightrag-db", "-l",
        type=Path,
        help="LightRAG存储目录 (默认: working/lightrag_db/[domain])"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="输出目录 (默认: working/output/[domain])"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        default=True,
        help="递归处理子目录"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="禁用问题去重"
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
        domains = list_available_domains(paths["lightrag_db_dir"])
        if domains:
            print("可用的 domain 目录 (已有知识图谱):")
            for d in domains:
                print(f"  - {d}")
        else:
            print(f"未找到 domain 目录 (在 {paths['lightrag_db_dir']} 下)")
            print("请先运行步骤2 (2_build_graph.py) 构建知识图谱")
        return

    # 确定路径
    if args.input:
        input_path = args.input
    elif args.domain:
        input_path = paths["processed_dir"] / args.domain
    else:
        input_path = paths["processed_dir"]

    if args.lightrag_db:
        lightrag_db = args.lightrag_db
    elif args.domain:
        lightrag_db = paths["lightrag_db_dir"] / args.domain
    else:
        lightrag_db = paths["lightrag_db_dir"]

    if args.output:
        output_path = args.output
    elif args.domain:
        output_path = paths["output_dir"] / args.domain
    else:
        output_path = paths["output_dir"]

    # 确保目录存在
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("步骤3: 生成问题")
    logger.info("=" * 60)
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"输入文本目录: {input_path}")
    logger.info(f"LightRAG存储: {lightrag_db}")
    logger.info(f"输出目录: {output_path}")

    # 检查路径
    if not input_path.exists():
        logger.error(f"输入路径不存在: {input_path}")
        sys.exit(1)

    if not lightrag_db.exists():
        logger.error(f"LightRAG存储不存在: {lightrag_db}")
        if args.domain:
            available = list_available_domains(paths["lightrag_db_dir"])
            if available:
                logger.info(f"可用的 domain: {', '.join(available)}")
            else:
                logger.info("请先运行步骤2 (2_build_graph.py) 构建知识图谱")
        sys.exit(1)

    # 查找文本文件
    pattern = "**/*.txt" if args.recursive else "*.txt"
    text_files = sorted(input_path.glob(pattern))

    if not text_files:
        logger.error(f"未找到文本文件: {input_path}")
        sys.exit(1)

    logger.info(f"发现 {len(text_files)} 个文本文件")

    # 初始化RAGCore（用于获取知识图谱上下文）
    logger.info("加载LightRAG知识库...")
    rag = RAGCore(
        working_dir=lightrag_db,
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

    # 初始化LLM客户端
    llm = LLMClient(
        base_url=settings.question_gen.llm.base_url,
        model=settings.question_gen.llm.model,
        max_tokens=settings.question_gen.llm.max_tokens,
        temperature=settings.question_gen.llm.temperature,
        timeout=settings.question_gen.llm.timeout,
        questions_per_chunk=settings.question_gen.questions_per_chunk,
        system_prompt=settings.prompts.system_prompt,
        human_prompt=settings.prompts.human_prompt,
        rag=rag,
        kg_context_config={
            "enabled": settings.question_gen.kg_context.enabled,
            "max_entities": settings.question_gen.kg_context.max_entities,
            "max_relations": settings.question_gen.kg_context.max_relations,
            "max_snippets": settings.question_gen.kg_context.max_snippets,
            "snippet_chars": settings.question_gen.kg_context.snippet_chars,
        },
    )

    # 初始化文本分块器
    chunker = TextChunker(
        use_token_chunking=settings.chunking.use_token_chunking,
        tokenizer_model=settings.chunking.tokenizer_model,
        chunk_token_size=settings.chunking.chunk_token_size,
        chunk_overlap_token_size=settings.chunking.chunk_overlap_token_size,
    )

    # 处理每个文本文件
    total_questions = 0
    processed_files = 0

    for text_file in tqdm(text_files, desc="生成问题"):
        try:
            logger.info(f"处理文件: {text_file.name}")

            # 读取文本
            content = text_file.read_text(encoding="utf-8")
            if not content.strip():
                logger.warning(f"空文件，跳过: {text_file.name}")
                continue

            # 分块
            document_id = text_file.stem
            chunks = chunker.chunk_text(content, document_id)

            if not chunks:
                logger.warning(f"未生成分块，跳过: {text_file.name}")
                continue

            logger.info(f"  分块数量: {len(chunks)}")

            # 生成问题
            question_set = llm.generate_questions_from_chunks(chunks)

            # 去重
            if not args.no_dedup and settings.question_gen.quality.enable_deduplication:
                original_count = len(question_set.questions)
                question_set.questions = deduplicate_questions(
                    question_set.questions,
                    settings.question_gen.quality.dedup_similarity_threshold
                )
                dedup_count = original_count - len(question_set.questions)
                if dedup_count > 0:
                    logger.info(f"  去重移除: {dedup_count} 个问题")

            # 保存问题
            output_file = output_path / f"{document_id}_questions.jsonl"
            save_questions_jsonl(question_set, output_file)

            total_questions += question_set.total_questions
            processed_files += 1

            logger.info(f"  生成问题: {question_set.total_questions} 个")

        except Exception as e:
            logger.error(f"处理文件失败 {text_file.name}: {e}")
            continue

    logger.info("=" * 60)
    logger.info("生成完成")
    if args.domain:
        logger.info(f"Domain: {args.domain}")
    logger.info(f"处理文件: {processed_files}/{len(text_files)}")
    logger.info(f"总问题数: {total_questions}")
    logger.info(f"输出目录: {output_path}")
    logger.info("=" * 60)

    logger.info("步骤3完成！")


if __name__ == "__main__":
    main()
