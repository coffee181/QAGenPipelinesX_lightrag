# 文档隔离问题修复方案

## 问题描述

当前实现中，所有文档都插入到同一个 `working_dir` 的知识库中，导致：
- 所有文档的向量数据混合在一起
- RAG检索时无法区分文档来源
- 为文档A生成答案时，可能检索到文档B、C的内容

## 核心问题代码

```python
# main.py:664
answer_service.setup_knowledge_base(input_path, working_dir)
# ❌ 所有文档都插入到同一个 working_dir！
```

## 解决方案A：文档专属知识库目录（推荐）⭐

### 实现思路

为每个文档创建独立的知识库子目录：

```
working_dir/
├── document_A_kb/    # 文档A专属知识库
│   ├── vdb_chunks.json
│   ├── vdb_entities.json
│   └── ...
├── document_B_kb/    # 文档B专属知识库
│   ├── vdb_chunks.json
│   ├── vdb_entities.json
│   └── ...
```

### 需要修改的地方

1. **`answer_service.py` - 添加按文档创建知识库的方法**

```python
def setup_knowledge_base_for_document(
    self,
    document_path: Path,
    base_working_dir: Path,
    document_id: str
) -> Path:
    """
    为单个文档创建专属知识库
    
    Returns:
        Path: 文档专属的 working_dir
    """
    # 创建文档专属目录
    doc_working_dir = base_working_dir / f"{document_id}_kb"
    doc_working_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置并插入文档
    self.rag.set_working_directory(doc_working_dir)
    document = self._load_document_from_file(document_path)
    self.rag.insert_document(document)
    
    return doc_working_dir
```

2. **`main.py` - 修改答案生成流程**

```python
def _generate_answers_from_questions(args, services, logger, session_id, question_results):
    """从问题结果生成答案"""
    _, _, answer_service, _ = services
    
    for question_result in question_results:
        document_id = question_result.document_id
        
        # 🔧 为每个文档创建专属知识库
        doc_working_dir = answer_service.setup_knowledge_base_for_document(
            document_path=input_path,  # 需要传递文档路径
            base_working_dir=PathUtils.normalize_path(args.working_dir),
            document_id=document_id
        )
        
        # 使用文档专属知识库生成答案
        questions_file = output_questions_dir / f"{document_id}_questions.jsonl"
        output_qa_file = output_dir / f"{document_id}_qapairs.jsonl"
        
        qa_result = answer_service.generate_answers_from_existing_kb(
            questions_file,
            doc_working_dir,  # 使用文档专属目录
            output_qa_file,
            f"{session_id}_answers_{document_id}",
            resume=True
        )
```

## 解决方案B：共享知识库 + 提示词过滤（已实现但不可靠）

已在 `lightrag_rag.py:536-543` 添加了文档过滤指令：

```python
if source_document:
    document_filter_instruction = f"""
⚠️ CRITICAL: DOCUMENT FILTERING ENABLED
You MUST ONLY use information from document: "{source_document}"
IGNORE all other documents in the knowledge base.
"""
```

**问题**：LightRAG 的向量检索已经返回了混合的文档块，仅靠提示词无法完全隔离。

## 推荐实施步骤

1. ✅ 已添加 `source_document` 参数到接口（临时缓解）
2. 🔧 实现方案A：文档专属知识库目录
3. 🧪 测试验证文档隔离效果
4. 📝 更新文档和使用说明

## 验证方法

生成QA对后，检查：
1. 答案中的型号、参数是否与问题来源文档一致
2. 不同文档的QA对不应出现交叉引用
3. 每个文档的知识库目录应该独立

## 注意事项

- 文档专属知识库会增加存储空间占用
- 但换来了完全的文档隔离和答案准确性
- 建议在测试完成后清理临时知识库目录

