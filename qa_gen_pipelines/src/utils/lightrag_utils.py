"""LightRAG 相关的公用工具函数。"""

from __future__ import annotations

import asyncio
import hashlib
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence


try:  # pragma: no cover - 依赖外部 LightRAG 安装
    from lightrag.utils import clean_text as _lightrag_clean_text  # type: ignore
    from lightrag.utils import compute_mdhash_id as _lightrag_compute_mdhash_id  # type: ignore
except Exception:  # pragma: no cover - 运行环境可能缺失
    _lightrag_clean_text = None
    _lightrag_compute_mdhash_id = None


def _fallback_clean_text(text: str) -> str:
    """退化实现：移除首尾空白以及空字节。"""
    return text.replace("\x00", "").strip()


def compute_lightrag_chunk_id(content: Optional[str]) -> Optional[str]:
    """
    按照 LightRAG 的规则为文本块计算 chunk_id。

    LightRAG 内部会先 clean_text（strip + 去掉空字节），再对结果做 MD5，并加 `chunk-` 前缀。
    这里尽可能复用官方实现，若库不可用则使用等价的降级逻辑。
    """
    if not content:
        return None

    cleaner = _lightrag_clean_text or _fallback_clean_text
    cleaned = cleaner(content)
    if not cleaned:
        return None

    if _lightrag_compute_mdhash_id:
        return _lightrag_compute_mdhash_id(cleaned, prefix="chunk-")

    return "chunk-" + hashlib.md5(cleaned.encode("utf-8")).hexdigest()


def build_chunk_citation(
    chunk_id: Optional[str],
    chunk_data: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """
    根据 LightRAG text_chunks 存储的数据构建 citation 信息。

    Args:
        chunk_id: chunk 的唯一标识
        chunk_data: 从 LightRAG text_chunks 读取到的详情

    Returns:
        citation 字典，或在缺少有效数据时返回 None
    """
    if not chunk_id:
        return None

    citation = {
        "chunk_id": chunk_id,
    }

    if chunk_data:
        citation.update(
            {
                "file_path": chunk_data.get("file_path", "unknown_source"),
                "full_doc_id": chunk_data.get("full_doc_id"),
                "chunk_order_index": chunk_data.get("chunk_order_index"),
                "tokens": chunk_data.get("tokens"),
            }
        )

        content = chunk_data.get("content")
        if isinstance(content, str):
            citation["preview"] = content[:200]

    return citation


def extract_chunk_ids_from_source(
    raw_source: Any,
    separators: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    将 LightRAG 节点或边上的 source_id 字段解析为 chunk_id 列表。
    """
    if not raw_source:
        return []

    if separators is None:
        separators = [";", ",", "|"]

    chunk_ids: List[str] = []

    if isinstance(raw_source, str):
        pattern = "|".join(re.escape(sep) for sep in separators if sep)
        parts = re.split(pattern, raw_source) if pattern else [raw_source]
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                chunk_ids.append(cleaned)
    elif isinstance(raw_source, (list, tuple, set)):
        for item in raw_source:
            if isinstance(item, str) and item.strip():
                chunk_ids.append(item.strip())

    return chunk_ids


class LightRAGContextBuilder:
    """用于构建知识图谱上下文的帮助类。"""

    def __init__(
        self,
        rag_impl: Any,
        *,
        max_entities: int = 3,
        max_relations: int = 2,
        max_snippets: int = 2,
        snippet_chars: int = 200,
        max_related_chunk_ids: int = 6,
    ) -> None:
        self.rag_impl = rag_impl
        self.max_entities = max_entities
        self.max_relations = max_relations
        self.max_snippets = max_snippets
        self.snippet_chars = snippet_chars
        self.max_related_chunk_ids = max_related_chunk_ids

    def build_context(self, chunk_id: Optional[str]) -> Dict[str, Any]:
        if not chunk_id or not self.rag_impl:
            return self._empty_context()

        rag_core = getattr(self.rag_impl, "rag", None)
        if not rag_core:
            return self._empty_context()

        graph = getattr(rag_core, "chunk_entity_relation_graph", None)
        text_chunks = getattr(rag_core, "text_chunks", None)
        if not graph:
            return self._empty_context()

        try:
            nodes = self._run_async(graph.get_nodes_by_chunk_ids([chunk_id])) or []
        except Exception:
            nodes = []

        try:
            edges = self._run_async(graph.get_edges_by_chunk_ids([chunk_id])) or []
        except Exception:
            edges = []

        entity_lines: List[str] = []
        relation_lines: List[str] = []
        snippet_lines: List[str] = []
        related_entities: List[str] = []
        related_chunk_ids: List[str] = []

        for node in nodes:
            if len(entity_lines) >= self.max_entities:
                break

            name = (
                node.get("entity_id")
                or node.get("entity_name")
                or node.get("id")
            )
            if not name or name in related_entities:
                continue

            entity_type = node.get("entity_type")
            description = node.get("description")
            info = f"- 实体 {name}"
            if entity_type:
                info += f"（类型：{entity_type}）"
            if description:
                info += f"：{description}"
            else:
                info += "：暂无详细描述"

            entity_lines.append(info)
            related_entities.append(name)

            for cid in extract_chunk_ids_from_source(node.get("source_id")):
                if (
                    cid
                    and cid not in related_chunk_ids
                    and cid != chunk_id
                    and len(related_chunk_ids) < self.max_related_chunk_ids
                ):
                    related_chunk_ids.append(cid)

        for edge in edges:
            if len(relation_lines) >= self.max_relations:
                break

            if "src_tgt" in edge:
                entity1, entity2 = edge.get("src_tgt", (None, None))
            else:
                entity1 = edge.get("src_id")
                entity2 = edge.get("tgt_id")

            if not entity1 or not entity2:
                continue

            description = edge.get("description")
            keywords = edge.get("keywords")

            info = f"- {entity1} ↔ {entity2}"
            if description:
                info += f"：{description}"
            elif keywords:
                info += f"：关键词 {keywords}"
            else:
                info += "：暂无详细描述"

            relation_lines.append(info)

            for candidate in (entity1, entity2):
                if candidate and candidate not in related_entities:
                    related_entities.append(candidate)

            for cid in extract_chunk_ids_from_source(edge.get("source_id")):
                if (
                    cid
                    and cid not in related_chunk_ids
                    and cid != chunk_id
                    and len(related_chunk_ids) < self.max_related_chunk_ids
                ):
                    related_chunk_ids.append(cid)

        if text_chunks and related_chunk_ids:
            for cid in related_chunk_ids[: self.max_snippets]:
                try:
                    data = self._run_async(text_chunks.get_by_id(cid))
                except Exception:
                    data = None

                if not data:
                    continue

                content = data.get("content", "")
                if content:
                    snippet = content.strip().replace("\n", " ")
                    snippet_lines.append(
                        f"- 片段 {cid}: {snippet[: self.snippet_chars]}"
                    )

        context_sections: List[str] = []
        if entity_lines:
            context_sections.append("【相关实体信息】\n" + "\n".join(entity_lines))
        if relation_lines:
            context_sections.append("【相关关系信息】\n" + "\n".join(relation_lines))
        if snippet_lines:
            context_sections.append("【知识图谱片段】\n" + "\n".join(snippet_lines))

        prompt_context = "\n\n".join(context_sections).strip()

        ordered_chunk_ids = [chunk_id]
        for cid in related_chunk_ids:
            if cid not in ordered_chunk_ids:
                ordered_chunk_ids.append(cid)

        return {
            "prompt_context": prompt_context,
            "related_entities": related_entities,
            "related_chunk_ids": ordered_chunk_ids,
        }

    def _run_async(self, coro: Any, timeout: float = 5.0) -> Any:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(asyncio.wait_for(coro, timeout=timeout))

    @staticmethod
    def _empty_context() -> Dict[str, Any]:
        return {
            "prompt_context": "",
            "related_entities": [],
            "related_chunk_ids": [],
        }
