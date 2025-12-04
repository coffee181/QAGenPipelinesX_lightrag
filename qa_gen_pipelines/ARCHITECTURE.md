# QAGenPipelinesX_lightrag 系统技术架构文档

## 目录

1. [系统概览](#系统概览)
2. [整体架构](#整体架构)
3. [核心组件详解](#核心组件详解)
4. [数据流与处理流程](#数据流与处理流程)
5. [存储结构](#存储结构)
6. [配置系统](#配置系统)
7. [关键算法与技术细节](#关键算法与技术细节)
8. [接口与扩展](#接口与扩展)

---

## 系统概览

### 项目简介

**QAGenPipelinesX_lightrag** 是一个基于知识图谱和向量检索的智能问答对生成系统。该系统专为技术文档(如产品手册、设备说明书、技术规格书)设计,能够自动从 PDF 文档中提取文本、构建知识图谱、生成高质量问答对,用于训练和评估问答系统。

### 核心功能

1. **文档预处理**: OCR 识别 PDF 文档,提取纯文本
2. **文本切分**: 基于 Token 的智能文本分块
3. **知识图谱构建**: 使用 LightRAG 提取实体和关系,构建知识图谱
4. **向量化索引**: 对文本块、实体、关系进行向量化并建立索引
5. **问题生成**: 基于文本块和知识图谱上下文生成问题
6. **答案生成**: 使用混合检索(知识图谱+向量搜索)生成答案
7. **质量控制**: 答案有效性验证、幻觉检测

### 技术栈

- **Python 3.10+**
- **LightRAG**: 知识图谱构建和检索
- **NanoVectorDB**: 轻量级向量数据库
- **NetworkX**: 图存储和操作
- **Tiktoken**: OpenAI 标准分词器
- **PaddleOCR**: 中文 OCR 识别
- **Ollama**: 本地 LLM 推理
- **Loguru**: 日志管理

---

## 整体架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                          QA Generation System                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐       ┌───────────────┐
│ Document      │         │ Knowledge     │       │ QA Pair       │
│ Processing    │────────▶│ Graph & Vector│──────▶│ Generation    │
│ Pipeline      │         │ Store         │       │ Pipeline      │
└───────────────┘         └───────────────┘       └───────────────┘
        │                         │                         │
        ▼                         ▼                         ▼
┌───────────────┐         ┌───────────────┐       ┌───────────────┐
│ - PDF OCR     │         │ - Entity Ext. │       │ - Question Gen│
│ - Text Clean  │         │ - Relation Ext│       │ - Answer Gen  │
│ - Chunk Split │         │ - Vector Index│       │ - Validation  │
└───────────────┘         └───────────────┘       └───────────────┘
```

### 三大处理流水线

#### 1. 文档预处理流水线

```
PDF 文件 → OCR 识别 → 文本清洗 → 纯文本文件(.txt)
```

#### 2. 知识图谱与向量化流水线

```
纯文本 → Token切分 → 实体抽取 → 关系抽取 → 知识图谱
   │                                              ↓
   └────────→ 向量化 → 实体向量库
                     → 关系向量库
                     → Chunk向量库
```

#### 3. 问答对生成流水线

```
纯文本 → 文本切分 → 问题生成(带KG上下文) → 问题文件(.jsonl)
                                              ↓
        知识图谱 + 向量库 ← 混合检索 ← 答案生成 ← 问题文件
                                              ↓
                                      答案验证 & 质量过滤
                                              ↓
                                      QA对文件(.jsonl)
```

---

## 核心组件详解

### 1. 文档处理模块 (Document Processing)

#### 1.1 PDF 预处理服务 (`PDFProcessor`)

**位置**: `src/services/pdf_processor.py`

**职责**:
- 接收 PDF 文档
- 调用 OCR 引擎识别文本
- 清理和格式化文本
- 输出纯文本文件

**关键流程**:

```python
def process_pdf(self, pdf_path: Path) -> Document:
    """
    1. 读取 PDF 文件
    2. 逐页调用 OCR 识别
    3. 合并识别结果
    4. 清理 markdown 格式
    5. 保存为 .txt 文件
    """
```

**OCR 实现**:

- **PaddleOCR** (`src/implementations/paddle_ocr.py`): 
  - 支持中英文混合识别
  - GPU 加速
  - 置信度阈值过滤
  
- **Tesseract OCR** (`src/implementations/tesseract_ocr.py`):
  - 备用 OCR 方案
  - 纯 CPU 运行

**输出示例**:
```
working/processed/01-GSK_27i高端多通道数控系统_20250226.txt
```

---

### 2. 文本切分模块 (Text Chunking)

#### 2.1 SimpleTextChunker

**位置**: `src/implementations/simple_text_chunker.py`

**核心功能**: 将长文本切分成固定大小的 chunk,支持三种切分策略

#### 2.1.1 Token 级切分 (推荐,与 LightRAG 一致)

**原理**:
- 使用 Tiktoken 分词器(o200k_base 或 cl100k_base)
- 按 token 数量切分,而不是字符数
- 保证每个 chunk 不超过 LLM 的 context window

**配置参数**:
```yaml
text_chunker:
  use_token_chunking: true
  tokenizer_model: "o200k_base"        # 分词器模型
  chunk_token_size: 1200               # 每个chunk的token数
  chunk_overlap_token_size: 100        # chunk间重叠的token数
```

**切分算法**:

```python
def _chunk_by_tokens(self, text: str, document_id: str) -> List[DocumentChunk]:
    """
    1. 使用 tiktoken 编码文本 → tokens
    2. 按固定 token 窗口滑动切分:
       - window_size = 1200 tokens
       - overlap = 100 tokens
       - step = window_size - overlap = 1100 tokens
    3. 每个 chunk 解码回文本
    4. 计算每个 chunk 的 LightRAG chunk_id (MD5哈希)
    """
```

**Chunk ID 计算**:

```python
def compute_lightrag_chunk_id(content: str) -> str:
    """
    使用与 LightRAG 完全一致的方式计算 chunk_id:
    MD5(content.encode('utf-8')).hexdigest()[:20]
    
    这样确保:
    - 问题生成时引用的 chunk_id
    - 答案生成时检索的 chunk_id
    - LightRAG 存储的 chunk_id
    三者完全一致
    """
```

#### 2.1.2 句子级切分

**原理**:
- 按句子边界切分(中英文)
- 保持句子完整性
- 尊重最大字符数限制

**适用场景**: 需要保持语义完整性的场景

#### 2.1.3 字符级切分

**原理**:
- 简单按字符数切分
- 固定窗口大小和重叠

**适用场景**: 快速切分,不考虑语义边界

---

### 3. Chunk 持久化仓库 (ChunkRepository)

**位置**: `src/utils/chunk_repository.py`

**职责**: 持久化存储切分后的 chunk,供后续问题生成和答案生成使用

**存储格式**: JSON 或 SQLite

#### JSON 存储结构

**文件路径**: `lightrag_cache/chunks/chunks.json`

**数据模型**:
```json
{
  "chunk-id-abc123": {
    "chunk_id": "chunk-id-abc123",
    "document_id": "path/to/doc.txt",
    "content": "实际chunk文本内容...",
    "tokens": 250,
    "chunk_order_index": 0,
    "total_chunks": 15,
    "start_position": 0,
    "end_position": 1250,
    "metadata": {}
  }
}
```

**关键方法**:
- `upsert_chunks()`: 批量插入或更新 chunk
- `get_chunk_by_id()`: 根据 chunk_id 获取单个 chunk
- `get_chunks_by_document()`: 获取某文档的所有 chunk
- `get_contents_by_ids()`: 批量获取 chunk 内容

---

### 4. 知识图谱与向量化模块 (LightRAG)

#### 4.1 LightRAG 集成 (`LightRAGImplementation`)

**位置**: `src/implementations/lightrag_rag.py`

**核心功能**: 封装 LightRAG,提供文档插入、知识图谱构建、混合检索

#### 4.1.1 工作目录结构

**默认路径**: `working/vectorized/`

**文件说明**:

```
vectorized/
├── graph_chunk_entity_relation.graphml     # 知识图谱(NetworkX GraphML格式)
├── vdb_entities.json                       # 实体向量数据库
├── vdb_relationships.json                  # 关系向量数据库
├── vdb_chunks.json                         # Chunk向量数据库
├── kv_store_full_docs.json                 # 完整文档KV存储
├── kv_store_text_chunks.json               # 文本chunk KV存储
├── kv_store_full_entities.json             # 实体详情KV存储
├── kv_store_full_relations.json            # 关系详情KV存储
├── kv_store_entity_chunks.json             # 实体-chunk映射
├── kv_store_relation_chunks.json           # 关系-chunk映射
├── kv_store_llm_response_cache.json        # LLM响应缓存
└── kv_store_doc_status.json                # 文档处理状态
```

#### 4.1.2 文档插入流程

**方法**: `insert_documents_to_working_dir()`

```python
async def insert_documents_to_working_dir(
    self,
    text_files: List[Path],
    working_dir: Path
) -> dict:
    """
    完整向量化流程:
    
    1. 设置工作目录 (set_working_directory)
       - 初始化 LightRAG 实例
       - 加载或创建知识图谱
       - 初始化向量数据库
    
    2. 逐文档处理:
       a. 读取文本内容
       b. 调用 LightRAG.ainsert(content)
       c. LightRAG 内部执行:
          - Token 切分 (chunking_by_token_size)
          - 实体抽取 (LLM 调用)
          - 关系抽取 (LLM 调用)
          - 实体合并去重
          - 关系合并去重 (应用补丁)
          - 向量化 (embedding)
          - 写入向量数据库
          - 更新知识图谱
    
    3. 返回状态统计
    """
```

**关键步骤详解**:

##### 实体抽取

LightRAG 使用 LLM 从每个 chunk 中抽取实体,输出格式:

```
实体名称<SEP>实体类型<SEP>实体描述<SEP>关键词1,关键词2
```

**示例**:
```
VMC850L<SEP>设备型号<SEP>立式加工中心型号<SEP>机床,加工中心,VMC
```

##### 关系抽取

从每个 chunk 抽取实体间的关系:

```
源实体<SEP>目标实体<SEP>关系类型<SEP>关系描述<SEP>关键词
```

**示例**:
```
VMC850L<SEP>BT40<SEP>使用<SEP>该机床使用BT40主轴接口<SEP>主轴,接口
```

##### 关系合并补丁

**问题**: LightRAG 原版在合并关系时可能丢失 description

**解决方案**: `src/utils/lightrag_relation_patch.py`

```python
async def patched_merge_edges_then_upsert(
    src_id: str,
    tgt_id: str,
    edges_data: list[dict],
    ...
) -> None:
    """
    补丁功能:
    1. 在调用原始 _merge_edges_then_upsert 之前
    2. 检查每个关系的 description 字段
    3. 如果缺失,填充默认描述: "关系: {src} -> {tgt}"
    4. 然后调用原始函数
    
    这确保所有关系都有有效描述,避免后续检索时出错
    """
```

**应用方式**:
```python
import lightrag.operate as lightrag_operate
lightrag_operate._merge_edges_then_upsert = patched_merge_edges_then_upsert
```

##### 向量化与索引

LightRAG 使用 NanoVectorDB 作为向量存储:

**向量维度**: 3072 (text-embedding-3-large)

**索引结构**:
```python
{
  "embedding_dim": 3072,
  "metric": "cosine",           # 余弦相似度
  "data": [
    {
      "id": "entity-id-1",
      "embedding": [0.123, -0.456, ...],  # 3072维向量
      "metadata": {...}
    }
  ]
}
```

**三个向量库**:
1. **vdb_entities.json**: 实体向量
   - ID: 实体名称
   - 向量: 实体名称+描述的 embedding
   
2. **vdb_relationships.json**: 关系向量
   - ID: `{src_id}-{relation_type}-{tgt_id}`
   - 向量: 关系描述的 embedding
   
3. **vdb_chunks.json**: Chunk 向量
   - ID: chunk_id (MD5 hash)
   - 向量: chunk 内容的 embedding

#### 4.1.3 知识图谱结构

**格式**: NetworkX GraphML

**节点类型**:
- **Entity**: 实体节点
  - 属性: `entity_name`, `entity_type`, `description`, `source_id` (chunk_id)

**边类型**:
- **Relation**: 实体间关系
  - 属性: `keywords`, `weight`, `source_id` (chunk_id), `description`

**图查询示例**:

```python
import networkx as nx

G = nx.read_graphml("graph_chunk_entity_relation.graphml")

# 查询与实体相关的邻居
neighbors = list(G.neighbors("VMC850L"))

# 查询两实体间的关系
edges = G.get_edge_data("VMC850L", "BT40")
```

---

### 5. 问题生成模块 (Question Generation)

#### 5.1 LocalQuestionGenerator

**位置**: `src/implementations/local_question_generator.py`

**核心功能**: 基于 Ollama 本地模型从 chunk 生成问题

#### 5.1.1 问题生成流程

```python
def generate_questions_from_chunk(self, chunk: DocumentChunk) -> List[Question]:
    """
    1. 构建知识图谱上下文 (可选)
       a. 计算 chunk 的 LightRAG chunk_id
       b. 查询该 chunk 关联的实体
       c. 查询这些实体的关系
       d. 查询相关的其他 chunks
       e. 组装成上下文字符串
    
    2. 组装提示词
       - System Prompt: 定义问题生成任务
       - Human Prompt: chunk内容 + KG上下文
    
    3. 调用 Ollama API
       - 模型: deepseek-r1:32b (可配置)
       - 温度: 0.7
       - 最大 tokens: 2048
    
    4. 解析响应
       - 提取问题列表
       - 清理 <think> 标签 (DeepSeek R1 特性)
       - 验证问题格式
    
    5. 构建 Question 对象
       - 填充 metadata (related_entities, related_chunk_ids)
       - 关联源 chunk
    """
```

#### 5.1.2 知识图谱上下文增强

**类**: `LightRAGContextBuilder`

**位置**: `src/utils/lightrag_utils.py`

**功能**: 从知识图谱中提取与当前 chunk 相关的上下文信息

```python
def build_context(self, chunk_id: str) -> dict:
    """
    1. 根据 chunk_id 查询关联的实体
       - 从 kv_store_entity_chunks.json 反向查找
    
    2. 查询实体的详细信息
       - 从 kv_store_full_entities.json 获取
    
    3. 查询实体间的关系
       - 从知识图谱 GraphML 中查询
    
    4. 查询相关的其他 chunks
       - 通过实体-chunk映射获取
    
    5. 组装上下文字符串:
       【知识图谱参考】
       实体: VMC850L (设备型号) - 立式加工中心型号
       实体: BT40 (接口类型) - 主轴接口标准
       
       关系: VMC850L -[使用]-> BT40
         描述: 该机床使用BT40主轴接口
       
       相关文档片段:
       - [chunk-abc123]: VMC850L是一款高性能...
    """
```

**配置参数**:
```yaml
question_generator:
  local:
    enable_kg_context: true
    max_context_entities: 3          # 最多提取3个实体
    max_context_relations: 2         # 最多提取2个关系
    max_context_snippets: 2          # 最多提取2个相关chunk
    context_snippet_chars: 200       # 每个snippet的字符数
    max_related_chunk_ids: 6         # 最多关联6个chunk ID
```

#### 5.1.3 问题格式解析

**支持的格式**:

**格式1: 纯问题列表** (推荐)
```
问题1: GSK 27i数控系统支持哪些通道配置？
问题2: 如何在GSK 27i中配置刀具补偿？
问题3: 该系统支持的最大轴数是多少？
```

**格式2: 问答对格式** (兼容)
```
问答对1:
问题: GSK 27i数控系统支持哪些通道配置？
答案: 支持1-8通道...

问答对2:
问题: 如何在GSK 27i中配置刀具补偿？
答案: 在系统设置菜单...
```

**解析算法**:
```python
def parse_questions_from_response(self, response: str) -> List[Question]:
    """
    1. 清理 <think></think> 标签
    2. 正则匹配"问题N:"模式
    3. 验证问题有效性:
       - 长度 > 15 字符
       - 包含问号
       - 不是标题行
       - 不是空问题
    4. 如果失败,尝试兼容旧格式(问答对)
    5. 如果还失败,fallback提取(按行查找问号)
    """
```

#### 5.1.4 Question 数据模型

```python
@dataclass
class Question:
    question_id: str              # UUID
    content: str                  # 问题文本
    source_document: str          # 源文档ID
    source_chunk_id: str          # 源chunk ID
    question_index: int           # 问题序号
    created_at: datetime          # 创建时间
    metadata: dict                # 元数据
    source_chunk_content: str     # 源chunk完整内容
    related_entities: List[str]   # 相关实体列表
```

**metadata 字段**:
```json
{
  "has_answer": false,
  "lightrag_chunk_id": "abc123...",
  "related_entities": ["VMC850L", "BT40"],
  "related_chunk_ids": ["abc123", "def456"],
  "knowledge_context_used": true
}
```

#### 5.1.5 问题持久化

**输出文件**: `working/questions/{document_name}_questions.jsonl`

**格式**: JSONL (每个问题一个 JSON 对象,多行格式化)

```json
{
  "question_id": "uuid-1234",
  "content": "GSK 27i数控系统支持哪些通道配置？",
  "source_document": "path/to/doc.txt",
  "source_chunk_id": "chunk-abc",
  "question_index": 1,
  "created_at": "2025-12-04T10:00:00",
  "metadata": {
    "lightrag_chunk_id": "abc123",
    "related_entities": ["GSK 27i", "数控系统"],
    "related_chunk_ids": ["abc123", "def456"],
    "knowledge_context_used": true
  },
  "source_chunk_content": "完整chunk内容..."
}

{
  "question_id": "uuid-5678",
  ...
}
```

---

### 6. 答案生成模块 (Answer Generation)

#### 6.1 AnswerService

**位置**: `src/services/answer_service.py`

**核心功能**: 读取问题文件,使用 LightRAG 检索答案,验证答案有效性

#### 6.1.1 答案生成流程

```python
async def generate_answers_for_question_file(
    self,
    question_file: Path
) -> Tuple[List[QAPair], dict]:
    """
    完整答案生成流程:
    
    1. 加载问题文件 (JSONL)
    2. 确定源文档ID并过滤
    3. 逐问题生成答案:
       a. 从问题中提取关键信息
       b. 构建检索查询
       c. 调用 LightRAG 检索
       d. 验证答案有效性
       e. 重试机制 (最多3次)
    4. 保存 QA 对到文件
    5. 返回统计信息
    """
```

#### 6.1.2 LightRAG 检索模式

LightRAG 支持四种检索模式,本系统主要使用 **mix** 模式:

##### Mix 模式 (混合检索)

**原理**: 同时使用知识图谱和向量检索,融合结果

**检索流程**:

```python
async def query_single_question(
    self,
    question_content: str,
    mode: str = "mix"
) -> Tuple[str, dict]:
    """
    Mix 模式检索:
    
    1. 关键词提取 (LLM调用)
       - 输入: 问题文本
       - 输出: 高层关键词 (用于图搜索)
              低层关键词 (用于向量搜索)
    
    2. 知识图谱检索 (KG Query)
       a. 实体检索:
          - 用高层关键词查询实体向量库
          - 返回 top_k=40 个相似实体
       
       b. 关系检索:
          - 用高层关键词查询关系向量库
          - 返回 top_k=40 个相似关系
       
       c. 图扩展:
          - 根据检索到的实体和关系
          - 在知识图谱中查询邻居节点
          - 构建局部子图
    
    3. 向量检索 (Vector Query / Naive)
       - 用低层关键词+完整问题
       - 查询 chunk 向量库
       - 返回 top_k=20 个相似 chunks
    
    4. 结果融合
       a. 合并实体、关系、chunks
       b. 去重
       c. 截断到最大长度
       d. Round-robin 合并保证多样性
    
    5. 上下文构建
       - 组装检索结果为文本
       - 提供给 LLM
    
    6. 答案生成 (LLM调用)
       - System Prompt + 上下文 + 问题
       - LLM 生成答案
    
    7. 返回答案和元数据
    """
```

**配置参数**:
```yaml
rag:
  mode: "mix"
  top_k: 40                    # 实体/关系检索数量
  chunk_top_k: 20              # chunk检索数量
  cosine_threshold: 0.2        # 相似度阈值
  max_token_for_text_unit: 4000
  max_token_for_local_context: 4000
  max_token_for_global_context: 4000
```

**其他检索模式**:

- **naive**: 纯向量检索,不使用知识图谱
- **local**: 局部图检索,关注实体邻居
- **global**: 全局图检索,关注社区结构

#### 6.1.3 文档过滤机制

**问题**: 在多文档场景下,避免跨文档检索造成答案错误

**解决方案**: 在检索前设置文档过滤器

```python
def _apply_document_filter(self, document_name: str):
    """
    1. 确定源文档ID
    2. 在 LightRAG 中设置过滤器:
       rag.set_query_document_filter(document_name)
    
    3. LightRAG 内部:
       - 检索实体时只返回 source_id 匹配的实体
       - 检索关系时只返回 source_id 匹配的关系
       - 检索 chunks 时只返回对应文档的 chunks
    
    这确保答案只来自问题的源文档
    """
```

#### 6.1.4 答案验证

**验证层次**:

##### 1. 格式验证

```python
def _is_valid_answer(self, answer_text: str) -> Tuple[bool, str]:
    """
    检查:
    - 长度 >= 20 字符
    - 不以问号结尾
    - 不是"I don't know"等无效回答
    - 不是纯英文的"no context"回答
    """
```

##### 2. 无信息检测

```python
def _is_no_info_answer(self, answer_text: str) -> bool:
    """
    检测常见的"无法回答"模式:
    - "Sorry, I'm not able to..."
    - "I don't have information..."
    - "The provided context does not..."
    - "抱歉,我无法..."
    - "根据提供的信息无法..."
    """
```

##### 3. 幻觉检测 (已调整为仅告警)

```python
def _verify_answer_authenticity(
    self,
    question: str,
    answer: str,
    source_chunk_content: str
) -> Tuple[bool, str]:
    """
    检测答案是否包含源文档中不存在的内容:
    
    1. 提取答案中的型号名称 (如 VMC850L, GSK27i)
    2. 提取答案中的关键数字
    3. 检查这些内容是否存在于源 chunk 中
    
    ⚠️ 注意: 此检测已改为仅记录 warning,
              不会阻止答案保存
              (避免误杀技术参数)
    """
```

#### 6.1.5 重试机制

```python
MAX_RETRIES = 3

for attempt in range(1, MAX_RETRIES + 1):
    try:
        answer_text = await self.rag.query_single_question(
            question_content,
            mode="mix"
        )
        
        is_valid, invalid_type = self._is_valid_answer(answer_text)
        
        if is_valid:
            break  # 成功
        else:
            if attempt < MAX_RETRIES:
                logger.warning(f"第 {attempt} 次尝试答案无效,准备重试...")
                continue
            else:
                logger.error(f"所有 {MAX_RETRIES} 次尝试都失败")
                # 跳过此问题
    
    except Exception as e:
        logger.error(f"第 {attempt} 次尝试异常: {e}")
        if attempt < MAX_RETRIES:
            continue
        else:
            # 彻底失败
```

#### 6.1.6 QAPair 数据模型

```python
@dataclass
class QAPair:
    qa_id: str
    question: str
    answer: str
    source_document: str
    source_chunk_id: str
    answer_chunk_ids: List[str]  # 答案来源的chunk IDs
    created_at: datetime
    metadata: dict
```

**metadata 字段**:
```json
{
  "question_id": "uuid-1234",
  "question_metadata": {...},
  "retrieval_mode": "mix",
  "retrieval_stats": {
    "entities_found": 5,
    "relations_found": 3,
    "chunks_found": 8
  },
  "validation_passed": true,
  "attempt_count": 1
}
```

#### 6.1.7 QA 对持久化

**输出文件**: `working/qa-pairs/{document_name}_qa_pairs.jsonl`

**格式**: JSONL

```json
{
  "qa_id": "qa-uuid-1234",
  "question": "GSK 27i数控系统支持哪些通道配置？",
  "answer": "GSK 27i数控系统支持1到8通道的多通道配置...",
  "source_document": "path/to/doc.txt",
  "source_chunk_id": "chunk-abc",
  "answer_chunk_ids": ["chunk-abc", "chunk-def"],
  "created_at": "2025-12-04T10:05:00",
  "metadata": {
    ...
  }
}
```

---

### 7. 进度管理模块 (Progress Manager)

**位置**: `src/services/progress_manager.py`

**功能**: 跨会话的进度跟踪和恢复

#### 7.1 核心概念

- **Session**: 一次完整的操作(如向量化一批文档)
- **Item**: 一个处理单元(如一个文档、一个问题)
- **Status**: pending / running / completed / failed

#### 7.2 数据结构

**文件**: `progress.json`

```json
{
  "sessions": {
    "insert_docs_20251204_172527": {
      "session_id": "insert_docs_20251204_172527",
      "operation_type": "insert_docs",
      "status": "completed",
      "total_items": 5,
      "completed_items": 5,
      "failed_items": 0,
      "start_time": "2025-12-04T17:25:27",
      "end_time": "2025-12-04T17:30:00",
      "metadata": {
        "working_dir": "working/vectorized"
      },
      "items": {
        "doc1.txt": {
          "status": "completed",
          "start_time": "...",
          "end_time": "...",
          "error": null
        }
      }
    }
  }
}
```

#### 7.3 关键方法

```python
class ProgressManager:
    def create_session(session_id, operation_type, total_items, metadata)
    def update_session_progress(session_id, item_id, success, error=None)
    def complete_session(session_id, final_status)
    def get_remaining_files(session_id, all_files) -> List[str]
    def get_session_stats(session_id) -> dict
```

#### 7.4 使用场景

**场景1: 向量化中断后恢复**

```python
# 首次运行
session_id = "insert_docs_20251204_172527"
pm.create_session(session_id, "insert_docs", total=10)

# 处理到第5个文档时程序崩溃

# 重新运行
remaining = pm.get_remaining_files(session_id, all_files)
# 返回: [doc6.txt, doc7.txt, ..., doc10.txt]
# 只处理剩余文档
```

**场景2: 问答对生成进度跟踪**

```python
session_id = "answer_gen_20251204_180000"
pm.create_session(session_id, "answer_generation", total=100)

for question in questions:
    try:
        answer = generate_answer(question)
        pm.update_session_progress(session_id, question.id, True)
    except Exception as e:
        pm.update_session_progress(session_id, question.id, False, str(e))

pm.complete_session(session_id, "completed")
```

---

### 8. 事件循环管理 (Event Loop)

**位置**: `src/utils/thread_event_loop.py`

**问题**: LightRAG 是异步库,但我们的主流程是同步的,如何桥接？

#### 8.1 ThreadEventLoop

```python
class ThreadEventLoop:
    """
    在独立线程中运行 asyncio 事件循环
    
    功能:
    1. 创建一个后台线程
    2. 在该线程中启动 asyncio.new_event_loop()
    3. 主线程通过 run_coroutine_threadsafe 提交异步任务
    4. 等待结果返回
    """
    
    def run_async(self, coro):
        """同步接口运行异步协程"""
        future = asyncio.run_coroutine_threadsafe(coro, self.loop)
        return future.result()
```

#### 8.2 在 LightRAG 中的应用

```python
class LightRAGImplementation:
    def __init__(self):
        self.loop_thread_id = None
        self._loop = None
    
    def _ensure_event_loop(self):
        """
        确保事件循环在正确的线程中:
        1. 检查当前线程ID
        2. 如果与上次不同,创建新循环
        3. 否则复用现有循环
        
        这解决了:
        - 跨线程 PriorityQueue 错误
        - Event loop is closed 错误
        """
        current_thread_id = threading.get_ident()
        if (self._loop is None or 
            self.loop_thread_id != current_thread_id):
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self._loop)
            self.loop_thread_id = current_thread_id
        return self._loop
    
    def _run_async(self, coro):
        """
        在当前循环中运行协程:
        1. 获取或创建循环
        2. 如果循环在运行,用 run_coroutine_threadsafe
        3. 否则用 loop.run_until_complete
        """
        loop = self._ensure_event_loop()
        if loop.is_running():
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result()
        else:
            return loop.run_until_complete(coro)
```

---

### 9. 配置管理 (Configuration)

**位置**: `src/utils/config.py`

**配置文件**: `config_local.yaml`

#### 9.1 配置结构

```yaml
# ========== 文件处理配置 ==========
file_processing:
  input_dir: "../working/raw"
  output_dir: "../working/processed"
  allowed_extensions: [".pdf", ".txt", ".md"]

# ========== OCR 配置 ==========
ocr:
  type: "paddle"
  paddle:
    use_gpu: true
    lang: "ch"
    det_db_box_thresh: 0.5

# ========== 文本切分配置 ==========
text_chunker:
  use_token_chunking: true
  tokenizer_model: "o200k_base"
  chunk_token_size: 1200
  chunk_overlap_token_size: 100
  persist_chunks: true
  chunk_store:
    type: "json"
    path: "./lightrag_cache/chunks"

# ========== RAG 配置 ==========
rag:
  type: "lightrag"
  cache_enabled: true
  cache_dir: "./lightrag_cache"
  mode: "mix"
  working_directory: "D:/llm-test/QAGenPipelinesX_lightrag/working"
  
  lightrag:
    embedding_func_max_async: 16
    embedding_batch_num: 32
    llm_model_func: "ollama_model_complete"
    llm_model_name: "deepseek-r1:32b"
    llm_model_max_async: 4
    llm_model_max_token_size: 32768
    embedding_func: "ollama_embedding"
    embedding_model_name: "bge-m3:latest"
    embedding_dim: 1024
    top_k: 40
    chunk_top_k: 20
    cosine_threshold: 0.2

# ========== 问题生成配置 ==========
question_generator:
  type: "local"
  enable_deduplication: true
  dedup_similarity_threshold: 0.85
  enable_quality_filter: true
  
  local:
    model_name: "deepseek-r1:32b"
    base_url: "http://localhost:11434"
    max_tokens: 2048
    temperature: 0.7
    timeout: 120
    questions_per_chunk: 10
    enable_kg_context: true
    max_context_entities: 3
    max_context_relations: 2

# ========== 提示词配置 ==========
prompts:
  system_prompt: |
    你是一个专业的问题生成助手...
  
  human_prompt: |
    基于以下文本生成{questions_per_chunk}个问题:
    {text}
  
  answer_prompt: |
    基于以下上下文回答问题...
```

#### 9.2 ConfigManager

```python
class ConfigManager:
    def __init__(self, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
    
    def get(self, key: str, default=None):
        """
        支持嵌套键访问:
        config.get("rag.lightrag.top_k")
        → self.config["rag"]["lightrag"]["top_k"]
        """
```

---

## 数据流与处理流程

### 完整流程图

```
┌──────────────┐
│  PDF 文档    │
└──────┬───────┘
       │
       ▼
┌──────────────────────────┐
│  1. 文档预处理            │
│  scripts/preprocess_pdfs.py │
└──────────┬───────────────┘
           │
           ▼
    ┌──────────────┐
    │ .txt 文件    │
    └──────┬───────┘
           │
           ├─────────────────────────────┐
           │                             │
           ▼                             ▼
┌──────────────────────┐      ┌──────────────────────┐
│  2. 向量化            │      │  3. 问题生成         │
│  scripts/             │      │  scripts/            │
│  vectorize_texts.py   │      │  generate_qa_pairs.py│
└──────────┬────────────┘      └──────────┬───────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────┐
│ 知识图谱+向量库      │      │ questions/*.jsonl    │
│ - graph.graphml      │      └──────────┬───────────┘
│ - vdb_*.json         │                 │
└──────────┬────────────┘                 │
           │                              │
           └──────────┬───────────────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │  4. 答案生成         │
           │  (在 generate_qa_    │
           │   pairs.py 中)       │
           └──────────┬───────────┘
                      │
                      ▼
           ┌──────────────────────┐
           │ qa-pairs/*.jsonl     │
           └──────────────────────┘
```

### 使用脚本

#### 脚本1: 文档预处理

```bash
python scripts/preprocess_pdfs.py \
  --input ../raw_pdfs \
  --output ../working/processed
```

**功能**: PDF OCR → 纯文本

#### 脚本2: 向量化

```bash
python scripts/vectorize_texts.py \
  --input ../working/processed \
  --output ../working/vectorized
```

**功能**: 文本 → 知识图谱 + 向量索引

#### 脚本3: 生成问答对

```bash
python scripts/generate_qa_pairs.py \
  --input ../working/processed \
  --vectors ../working/vectorized \
  --output ../working/qa-pairs \
  --questions-output ../working/questions
```

**功能**: 
1. 生成问题 → `working/questions/*.jsonl`
2. 生成答案 → `working/qa-pairs/*.jsonl`

---

## 关键算法与技术细节

### 1. Chunk ID 一致性算法

**挑战**: 确保以下三个地方的 chunk_id 完全一致:
1. ChunkRepository 中存储的 chunk_id
2. Question 元数据中的 lightrag_chunk_id
3. LightRAG 知识图谱中的 source_id

**解决方案**:

```python
def compute_lightrag_chunk_id(content: str) -> str:
    """
    统一的 chunk_id 计算函数:
    
    1. 对 content 进行 UTF-8 编码
    2. 计算 MD5 哈希
    3. 取前20位十六进制字符
    
    这与 LightRAG 内部的实现完全一致
    """
    if not content:
        return ""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:20]
```

**应用点**:
- `SimpleTextChunker._chunk_by_tokens()`: 切分时计算
- `LocalQuestionGenerator._build_context_for_chunk()`: 查询时使用
- `LightRAGContextBuilder.build_context()`: 上下文构建时使用

### 2. 知识图谱查询优化

**问题**: 直接在大图上查询效率低

**优化策略**:

#### 2.1 向量召回预过滤

```python
# 不直接在图上遍历所有节点
# 而是先用向量检索召回 top_k 个候选实体
entity_candidates = vdb_entities.query(
    embedding=question_embedding,
    top_k=40,
    threshold=0.2
)

# 然后只在这些候选实体的子图上查询
subgraph = G.subgraph(entity_candidates)
```

#### 2.2 关系权重排序

```python
# 关系按权重排序(权重 = 出现频率)
relations = sorted(
    relations,
    key=lambda r: r.get("weight", 1),
    reverse=True
)[:max_relations]
```

#### 2.3 Chunk 去重

```python
# Round-robin 合并避免重复
def round_robin_merge(chunks_list):
    """
    从多个来源的 chunks 中轮流抽取,
    保证多样性,避免某个来源dominate
    """
    result = []
    while any(chunks_list):
        for chunks in chunks_list:
            if chunks:
                result.append(chunks.pop(0))
    return result
```

### 3. 答案质量控制流程

```
生成答案
   │
   ▼
格式验证 ──不通过──▶ 重试 (最多3次)
   │
通过
   ▼
无信息检测 ──是──▶ 重试
   │
不是
   ▼
幻觉检测 ──可疑──▶ 记录warning (不阻止)
   │
   ▼
保存答案
```

### 4. LLM 调用优化

#### 4.1 响应缓存

```python
# LightRAG 内置缓存
# kv_store_llm_response_cache.json

cache_key = f"{mode}:{operation}:{hash(prompt)}"
if cache_key in cache:
    return cache[cache_key]

response = llm_call(prompt)
cache[cache_key] = response
return response
```

#### 4.2 并发控制

```python
# 限制并发 LLM 调用
llm_model_max_async: 4      # 最多4个并发
embedding_func_max_async: 16 # embedding可以更多
```

#### 4.3 超时管理

```python
# Ollama 调用超时配置
def configure_ollama_timeout():
    return {
        'timeout': (60, 30000)  # (连接超时, 读取超时)
    }
```

---

## 接口与扩展

### 1. 核心接口设计

#### 1.1 RAG 接口

```python
class RAGInterface(ABC):
    @abstractmethod
    async def insert(self, content: str, **kwargs):
        """插入文档"""
        pass
    
    @abstractmethod
    async def query(self, query: str, mode: str, **kwargs) -> str:
        """检索并生成答案"""
        pass
    
    @abstractmethod
    def set_query_document_filter(self, document_name: str):
        """设置文档过滤器"""
        pass
```

**实现**:
- `LightRAGImplementation`: 基于 LightRAG
- `SimpleRAGImplementation`: 简单向量检索(无KG)

#### 1.2 问题生成接口

```python
class QuestionGeneratorInterface(ABC):
    @abstractmethod
    def generate_questions_from_chunk(
        self,
        chunk: DocumentChunk
    ) -> List[Question]:
        """从chunk生成问题"""
        pass
    
    @abstractmethod
    def validate_questions(
        self,
        questions: List[Question]
    ) -> bool:
        """验证问题质量"""
        pass
```

**实现**:
- `LocalQuestionGenerator`: 基于 Ollama 本地模型

#### 1.3 文本切分接口

```python
class TextChunkerInterface(ABC):
    @abstractmethod
    def chunk_text(
        self,
        text: str,
        document_id: str
    ) -> List[DocumentChunk]:
        """切分文本"""
        pass
```

**实现**:
- `SimpleTextChunker`: 支持 Token/句子/字符三种切分

### 2. 扩展点

#### 2.1 添加新的 RAG 实现

```python
class MyCustomRAG(RAGInterface):
    async def insert(self, content: str, **kwargs):
        # 自定义向量化逻辑
        pass
    
    async def query(self, query: str, mode: str, **kwargs):
        # 自定义检索逻辑
        pass

# 在 config.yaml 中配置
rag:
  type: "my_custom"
```

#### 2.2 添加新的 LLM 后端

```python
# 修改 LightRAG 的 llm_func
async def my_llm_func(prompt, **kwargs):
    # 调用你的 LLM API
    response = await my_llm_api.call(prompt)
    return response

# 配置
lightrag:
  llm_model_func: "my_llm_func"
```

#### 2.3 自定义答案验证逻辑

```python
# 在 AnswerService 中添加新的验证方法
def _custom_validation(self, answer: str, question: str) -> bool:
    # 你的验证逻辑
    return True

# 在 generate_answers_for_question_file 中调用
if not self._custom_validation(answer, question):
    continue
```

---

## 附录

### A. 常见问题排查

#### 问题1: 向量化失败

**症状**: `kv_store_doc_status.json` 中状态为 "failed"

**排查步骤**:
1. 检查日志中的 "LLM output format error"
2. 确认 LightRAG 关系补丁已应用
3. 检查 Ollama 服务是否正常
4. 查看 `llm_response_cache` 中的原始响应

**解决方案**:
- 更新 `lightrag_relation_patch.py` 以匹配最新 LightRAG 版本
- 调整实体/关系抽取的 prompt
- 增加 LLM timeout

#### 问题2: Mix 模式查不到结果

**症状**: "Raw search results: 0 entities, 0 relations, N chunks"

**原因**:
- 知识图谱为空(向量化未成功)
- 文档过滤器设置错误
- chunk_id 不一致

**排查**:
```python
# 检查图谱
import networkx as nx
G = nx.read_graphml("working/vectorized/graph_chunk_entity_relation.graphml")
print(f"节点数: {G.number_of_nodes()}")
print(f"边数: {G.number_of_edges()}")

# 检查向量库
import json
with open("working/vectorized/vdb_entities.json") as f:
    vdb = json.load(f)
    print(f"实体数: {len(vdb['data'])}")
```

#### 问题3: 答案被误判为幻觉

**症状**: 日志中出现 "检测到虚构型号"

**解决方案**:
- 已调整为仅 warning,不阻止保存
- 如需调整阈值,修改 `_verify_answer_authenticity`

### B. 性能优化建议

#### 1. 向量化加速

```yaml
lightrag:
  embedding_func_max_async: 32   # 增加并发
  embedding_batch_num: 64        # 增加批大小
```

#### 2. 内存优化

```yaml
lightrag:
  max_token_for_text_unit: 2000    # 减小上下文长度
  top_k: 20                         # 减少召回数量
```

#### 3. 缓存策略

```python
# 启用所有缓存
rag:
  cache_enabled: true

text_chunker:
  persist_chunks: true
```

### C. 数据统计示例

**一个完整流程的典型数据量**:

```
输入:
- 5 个 PDF 文档 (共200页)

文档预处理:
- 输出: 5 个 .txt 文件 (共50MB文本)

向量化:
- Chunks: 1250 个
- 实体: 350 个
- 关系: 520 个
- 向量化时间: 约45分钟 (deepseek-r1:32b)

问题生成:
- 总问题数: 12500 个 (10问题/chunk)
- 去重后: 8000 个
- 生成时间: 约2小时

答案生成:
- 成功答案: 7200 个 (90%)
- 无效答案: 800 个 (10%)
- 生成时间: 约3小时
- 平均检索时间: 1.5秒/问题

最终输出:
- QA 对: 7200 个
- 文件大小: 约80MB (JSONL)
```

---

**文档版本**: v1.0  
**最后更新**: 2025-12-04  
**作者**: QAGenPipelinesX_lightrag Team

