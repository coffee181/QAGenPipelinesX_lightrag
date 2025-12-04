"""
LightRAG 关系描述补丁

问题：LightRAG 在处理关系（Relations）时，要求每个关系都有描述（description）。
      但在处理中文文档时，有时 LLM 可能无法生成关系描述，导致报错：
      ValueError: Relation xxx~yyy has no description

解决方案：在 LightRAG 的关系合并逻辑中注入补丁，为空描述自动生成默认值
"""

import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def patch_lightrag_relation_merge():
    """
    为 LightRAG 的 operate.py 添加补丁，处理缺失关系描述的问题
    """
    try:
        from lightrag import operate
        
        # 保存原始函数
        original_merge_edges = operate._merge_edges_then_upsert
        
        async def patched_merge_edges_then_upsert(
            src_id: str,
            tgt_id: str,
            edges_data: list[Dict[str, Any]],
            knowledge_graph_inst,
            relationships_vdb=None,
            entity_vdb=None,
            global_config=None,
            pipeline_status=None,
            pipeline_status_lock=None,
            llm_response_cache=None,
            added_entities=None,
            relation_chunks_storage=None,
            entity_chunks_storage=None,
        ):
            """
            修补后的 _merge_edges_then_upsert 函数
            为缺失的关系描述添加默认值
            """
            sanitized_edges: list[Dict[str, Any]] = []
            for edge_data in edges_data:
                if "description" not in edge_data or not edge_data.get("description"):
                    entity1 = src_id.replace("_", " ").strip()
                    entity2 = tgt_id.replace("_", " ").strip()
                    edge_data["description"] = f"{entity1} 与 {entity2} 的关系"
                    logger.warning(
                        f"关系 {src_id}~{tgt_id} 缺少描述，使用默认值: '{edge_data['description']}'"
                    )
                sanitized_edges.append(edge_data)
            
            # 调用原始函数
            return await original_merge_edges(
                src_id=src_id,
                tgt_id=tgt_id,
                edges_data=sanitized_edges,
                knowledge_graph_inst=knowledge_graph_inst,
                relationships_vdb=relationships_vdb,
                entity_vdb=entity_vdb,
                global_config=global_config,
                pipeline_status=pipeline_status,
                pipeline_status_lock=pipeline_status_lock,
                llm_response_cache=llm_response_cache,
                added_entities=added_entities,
                relation_chunks_storage=relation_chunks_storage,
                entity_chunks_storage=entity_chunks_storage,
            )
        
        # 应用补丁
        operate._merge_edges_then_upsert = patched_merge_edges_then_upsert
        logger.info("✅ LightRAG 关系描述补丁已应用")
        return True
        
    except ImportError as e:
        logger.error(f"无法导入 LightRAG: {e}")
        return False
    except Exception as e:
        logger.error(f"补丁应用失败: {e}")
        return False


def generate_relation_description(src_id: str, tgt_id: str, relation_type: Optional[str] = None) -> str:
    """
    为关系生成默认描述
    
    Args:
        src_id: 源实体 ID
        tgt_id: 目标实体 ID
        relation_type: 关系类型（可选）
        
    Returns:
        关系描述文本
    """
    entity1 = src_id.replace("_", " ").strip()
    entity2 = tgt_id.replace("_", " ").strip()
    
    if relation_type:
        return f"{entity1} 通过 {relation_type} 与 {entity2} 相关联"
    else:
        return f"{entity1} 与 {entity2} 的关系"
