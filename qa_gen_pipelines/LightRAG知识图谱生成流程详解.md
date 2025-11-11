# LightRAG 知识图谱生成流程详解

## 📋 概述

LightRAG 是一个基于大语言模型的知识图谱构建和检索系统，它能够从文档中自动提取实体、关系和知识，构建结构化的知识图谱，并支持高效的语义检索。

## 🔧 核心组件架构

### 1. LightRAG 实例初始化

```python
# src/implementations/lightrag_rag.py:163-335
def _create_lightrag_instance(self):
    """Create LightRAG instance with proper configuration."""
    
    # 1. 定义异步LLM函数
    async def llm_model_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        """LLM function for LightRAG using local Ollama model."""
        # 使用本地Ollama模型 (deepseek-r1:32b)
        ollama_url = "http://localhost:11434/api/generate"
        model_name = "deepseek-r1:32b"
        
        # 构建完整提示词
        full_prompt = ""
        if system_prompt:
            full_prompt += f"{system_prompt}\n\n"
        
        # 处理历史消息
        if history_messages:
            # 格式化历史对话
            for msg in history_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        full_prompt += f"System: {content}\n\n"
                    elif role == "user":
                        full_prompt += f"User: {content}\n\n"
                    elif role == "assistant":
                        full_prompt += f"Assistant: {content}\n\n"
        
        # 添加当前提示
        full_prompt += f"User: {prompt}\n\nAssistant:"
        
        # 准备Ollama请求
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "num_predict": kwargs.get("max_tokens", 2048)
            }
        }
        
        # 重试机制 (最多5次)
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        ollama_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=1800)  # 30分钟超时
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            raw_response = result.get("response", "")
                            # 清理<think>标签
                            cleaned_response = self._clean_think_tags(raw_response)
                            return cleaned_response
                        else:
                            # 错误处理和重试逻辑
                            error_text = await response.text()
                            logger.error(f"Ollama API error {response.status}: {error_text}")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 2
                                continue
                            else:
                                raise Exception(f"Ollama API error {response.status}: {error_text}")
                                
            except asyncio.TimeoutError:
                logger.error(f"Ollama API timeout on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise Exception("Ollama API timeout after all retries")
                    
            except Exception as e:
                logger.error(f"Unexpected error in Ollama LLM function: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                else:
                    raise

    # 2. 定义异步嵌入函数
    async def embedding_func(texts: List[str]):
        """Embedding function for LightRAG."""
        import numpy as np

        if self.openai_api_key:
            try:
                import openai
                # 使用OpenAI嵌入
                client = openai.AsyncOpenAI(api_key=self.openai_api_key)
                response = await client.embeddings.create(
                    model="text-embedding-3-large",  # 使用大模型，3072维
                    input=texts
                )
                embeddings = [data.embedding for data in response.data]
                return np.array(embeddings)
            except Exception as e:
                logger.warning(f"OpenAI embedding failed: {e}, using fallback")

        # 简单回退 - 创建3072维嵌入
        import hashlib
        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            embedding = [(hash_int >> i) & 1 for i in range(3072)]  # 3072维
            embeddings.append(embedding)
        return np.array(embeddings, dtype=np.float32)

    # 3. 创建LightRAG实例
    try:
        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,  # 匹配现有知识库
                max_token_size=8192,
                func=embedding_func
            ),
            # 使用兼容编码
            encoding_model="cl100k_base"  # 使用cl100k_base而不是o200k_base
        )
    except TypeError:
        # 如果encoding_model参数不支持，尝试不使用它
        rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=llm_model_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=3072,  # 匹配现有知识库
                max_token_size=8192,
                func=embedding_func
            )
        )

    # 4. 初始化存储
    try:
        async def initialize_all():
            await rag.initialize_storages()
            try:
                from lightrag.kg.shared_storage import initialize_pipeline_status
                await initialize_pipeline_status()
            except ImportError:
                pass

        asyncio.run(initialize_all())
    except Exception as e:
        logger.error(f"FATAL: Failed to initialize LightRAG storages: {e}")
        raise RAGError(f"Failed to initialize LightRAG storages: {e}")

    return rag
```

## 🔄 知识图谱生成流程

### 第1步：文档插入准备

```python
# src/implementations/lightrag_rag.py:359-378
def insert_document(self, document: Document) -> None:
    """
    Insert a single document into the knowledge base.
    
    Args:
        document: Document to insert
    
    Raises:
        RAGError: If insertion fails
    """
    try:
        logger.info(f"Inserting document: {document.name}")
        
        # 使用异步辅助函数
        asyncio.run(self._async_insert_document(document))
        
        logger.info(f"Successfully inserted document: {document.name}")
        
    except Exception as e:
        raise RAGError(f"Failed to insert document {document.name}: {e}")
```

### 第2步：异步文档处理

```python
# src/implementations/lightrag_rag.py:380-408
async def _async_insert_document(self, document: Document) -> None:
    """
    Async helper for inserting documents.
    
    Args:
        document: Document to insert
    """
    # 确保存储已初始化
    try:
        await self.rag.initialize_storages()
        
        # 如果可用，初始化管道状态
        try:
            from lightrag.kg.shared_storage import initialize_pipeline_status
            await initialize_pipeline_status()
        except ImportError:
            pass  # 并非所有版本都可用
    except Exception as e:
        logger.warning(f"Storage initialization warning: {e}")
    
    # 插入文档
    try:
        await self.rag.ainsert(document.content)
    except Exception as e:
        if "history_messages" in str(e):
            logger.warning(f"LightRAG history_messages issue, this is a known problem with current version")
            raise RAGError(f"LightRAG version issue: {e}")
        else:
            raise e
```

## 🧠 LightRAG 内部知识图谱构建机制

### 1. 文本分块与预处理

LightRAG 内部会将文档内容进行智能分块：

```python
# LightRAG 内部处理流程（简化版）
def process_document_content(content: str):
    """
    LightRAG 内部文档处理流程
    """
    # 1. 文本分块
    chunks = split_text_into_chunks(content, chunk_size=1000)
    
    # 2. 实体提取
    entities = extract_entities_from_chunks(chunks)
    
    # 3. 关系识别
    relations = identify_relations_between_entities(entities)
    
    # 4. 知识图谱构建
    knowledge_graph = build_knowledge_graph(entities, relations)
    
    return knowledge_graph
```

### 2. 实体提取过程

```python
# LightRAG 实体提取机制（概念性）
async def extract_entities_from_text(text: str, llm_func):
    """
    使用LLM从文本中提取实体
    """
    entity_extraction_prompt = """
    从以下文本中提取所有重要的实体（人物、地点、组织、概念、技术术语等）：
    
    文本：{text}
    
    请以JSON格式返回实体列表，包含：
    - entity_name: 实体名称
    - entity_type: 实体类型
    - description: 实体描述
    - context: 出现上下文
    """
    
    response = await llm_func(entity_extraction_prompt.format(text=text))
    entities = parse_entity_response(response)
    return entities
```

### 3. 关系识别过程

```python
# LightRAG 关系识别机制（概念性）
async def identify_relations_between_entities(entities: List[Entity], llm_func):
    """
    识别实体之间的关系
    """
    relation_extraction_prompt = """
    分析以下实体之间的关系：
    
    实体列表：{entities}
    
    请识别实体之间的关系，包括：
    - 关系类型（属于、包含、影响、依赖等）
    - 关系强度
    - 关系描述
    
    以JSON格式返回关系列表。
    """
    
    response = await llm_func(relation_extraction_prompt.format(entities=entities))
    relations = parse_relation_response(response)
    return relations
```

### 4. 知识图谱存储结构

LightRAG 将知识图谱存储为以下结构：

```python
# LightRAG 知识图谱存储结构
knowledge_graph_structure = {
    "entities": {
        "entity_id": {
            "name": "实体名称",
            "type": "实体类型",
            "description": "实体描述",
            "properties": {
                "属性1": "值1",
                "属性2": "值2"
            },
            "embeddings": [0.1, 0.2, ...],  # 3072维向量
            "created_at": "2024-01-01T00:00:00"
        }
    },
    "relations": {
        "relation_id": {
            "source_entity": "源实体ID",
            "target_entity": "目标实体ID",
            "relation_type": "关系类型",
            "description": "关系描述",
            "confidence": 0.95,
            "created_at": "2024-01-01T00:00:00"
        }
    },
    "documents": {
        "doc_id": {
            "content": "文档内容",
            "chunks": ["块1", "块2", ...],
            "entities": ["实体ID1", "实体ID2", ...],
            "created_at": "2024-01-01T00:00:00"
        }
    }
}
```

## 🔍 知识图谱查询机制

### 1. 查询模式

LightRAG 支持多种查询模式：

```python
# src/implementations/lightrag_rag.py:486-614
def query_single_question(self, question: str) -> str:
    """
    Query the knowledge base with a single question.
    """
    try:
        logger.info(f"Querying question: {question[:100]}...")
        
        # 检查缓存
        if self.enable_cache:
            cached_response = self._check_cache(question)
            if cached_response:
                self.cache_hits += 1
                logger.info(f"✅ Cache hit! (total hits: {self.cache_hits}, misses: {self.cache_misses})")
                return cached_response
            else:
                self.cache_misses += 1
        
        # 使用现有事件循环
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Event loop is closed")
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        response = None
        
        # 使用naive模式获得最佳结果
        try:
            logger.info("Using naive mode for reliable results...")
            
            # 自定义提示词，要求简洁直接的答案
            custom_prompt = """---Role---
You are a professional technical documentation assistant. Your role is to provide direct, concise answers based on the provided knowledge base.

---Goal---
Generate a DIRECT and CONCISE answer to the user's question based ONLY on the Knowledge Base provided below.

CRITICAL REQUIREMENTS:
1. Answer DIRECTLY without any preambles like "根据提供的知识库" or "Based on the provided information"
2. Do NOT include reasoning process or explanation of how you found the answer
3. Do NOT mention the knowledge base, documents, or data sources in your answer
4. Simply state the facts as if you're directly reading from a manual
5. Keep the answer focused and to-the-point

---Conversation History---
{history}

---Knowledge Graph and Document Chunks---
{context_data}

---Response Rules---
- Answer format: DIRECT and CONCISE, as if reading from the source document
- NO preambles like "根据", "基于", "According to", "Based on"
- NO meta-commentary about the knowledge base or your search process
- Use the same language as the user's question
- If the answer involves numbers, specifications, or technical details, state them directly
- If you don't have the information, simply say you cannot find the specific information
- Do NOT make up any information not in the Knowledge Base

Response:"""
            
            # 添加超时防止挂起查询
            response = loop.run_until_complete(
                asyncio.wait_for(
                    self.rag.aquery(question, param=QueryParam(mode="naive"), system_prompt=custom_prompt),
                    timeout=30.0  # 30秒超时
                )
            )
            logger.info("Query completed with naive mode")
            
        except asyncio.TimeoutError:
            logger.warning("Query timed out after 30 seconds")
            response = "查询超时，请稍后重试或简化问题。"
            
        except Exception as e:
            logger.warning(f"Naive mode failed: {e}")
            # 尝试local模式作为最后手段
            try:
                logger.info("Trying local mode as fallback...")
                response = loop.run_until_complete(
                    asyncio.wait_for(
                        self.rag.aquery(question, param=QueryParam(mode="local"), system_prompt=custom_prompt),
                        timeout=15.0  # 更短的超时时间
                    )
                )
                logger.info("Query completed with local mode")
            except Exception:
                response = "抱歉，无法从知识库中找到相关信息来回答这个问题。"
        
        if response is None:
            response = "抱歉，查询过程中出现问题，无法生成答案。"
        
        logger.info(f"Generated answer: {len(response)} characters")
        
        # 如果启用缓存，缓存响应
        if self.enable_cache:
            self._update_cache(question, response)
        
        return response
        
    except Exception as e:
        raise RAGError(f"Failed to query question: {e}")
```

### 2. 查询模式详解

LightRAG 支持三种查询模式：

#### **Naive 模式**（推荐）
- **特点**：直接基于文档块进行检索和回答
- **优势**：结果可靠，速度快
- **适用场景**：大多数查询场景

#### **Local 模式**（备用）
- **特点**：基于本地知识图谱进行查询
- **优势**：能够利用实体关系
- **适用场景**：需要关系推理的复杂查询

#### **Global 模式**（当前版本有问题）
- **特点**：基于全局知识图谱进行查询
- **问题**：当前版本存在兼容性问题
- **状态**：暂时禁用

## 📊 知识图谱存储与缓存

### 1. 存储结构

```python
# LightRAG 工作目录结构
working_directory/
├── entities.json          # 实体存储
├── relations.json         # 关系存储
├── documents.json         # 文档存储
├── chunks.json           # 文档块存储
├── embeddings.json       # 嵌入向量存储
├── knowledge_graph.graphml # 知识图谱文件
└── pipeline_status.json  # 管道状态
```

### 2. 缓存机制

```python
# src/implementations/lightrag_rag.py:808-850
def _check_cache(self, question: str) -> Optional[str]:
    """
    Check if a similar question exists in cache.
    """
    normalized = self._normalize_question(question)
    
    # 首先检查完全匹配
    if normalized in self.retrieval_cache:
        return self.retrieval_cache[normalized]
    
    # 检查相似问题
    for cached_question, cached_response in self.retrieval_cache.items():
        similarity = self._calculate_question_similarity(normalized, cached_question)
        if similarity >= self.cache_similarity_threshold:
            logger.debug(f"Found similar cached question (similarity={similarity:.2f})")
            return cached_response
    
    return None

def _update_cache(self, question: str, response: str) -> None:
    """
    Update cache with new question-response pair.
    """
    normalized = self._normalize_question(question)
    self.retrieval_cache[normalized] = response
    
    # 限制缓存大小防止内存问题
    max_cache_size = 1000
    if len(self.retrieval_cache) > max_cache_size:
        # 移除最旧的条目（FIFO）
        oldest_key = next(iter(self.retrieval_cache))
        del self.retrieval_cache[oldest_key]
        logger.debug(f"Cache size limit reached, removed oldest entry")

def _calculate_question_similarity(self, text1: str, text2: str) -> float:
    """
    Calculate similarity between two questions using character n-grams.
    """
    def get_ngrams(text, n=2):
        return set(text[i:i+n] for i in range(len(text)-n+1))
    
    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    
    return len(intersection) / len(union) if union else 0.0
```

## 🚀 批量处理机制

### 1. 批量文档插入

```python
# src/implementations/lightrag_rag.py:410-433
def insert_documents_batch(self, documents: List[Document]) -> None:
    """
    Insert multiple documents into the knowledge base.
    """
    try:
        logger.info(f"Inserting {len(documents)} documents in batch")
        
        for document in tqdm(documents, desc="Inserting documents"):
            try:
                self.insert_document(document)
            except Exception as e:
                logger.error(f"Failed to insert document {document.name}: {e}")
                continue
        
        logger.info(f"Batch insertion completed")
        
    except Exception as e:
        raise RAGError(f"Batch insertion failed: {e}")
```

### 2. 目录批量处理

```python
# src/implementations/lightrag_rag.py:435-484
def insert_from_directory(self, directory_path: Path) -> None:
    """
    Insert all text files from a directory.
    """
    try:
        if not directory_path.exists():
            raise RAGError(f"Directory does not exist: {directory_path}")
        
        # 查找所有文本文件
        text_files = list(directory_path.glob("*.txt"))
        
        if not text_files:
            logger.warning(f"No text files found in directory: {directory_path}")
            return
        
        logger.info(f"Found {len(text_files)} text files in directory: {directory_path}")
        
        for text_file in tqdm(text_files, desc="Processing text files"):
            try:
                # 读取文件内容
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 创建文档对象
                document = Document(
                    file_path=text_file,
                    content=content,
                    file_type=text_file.suffix,
                    file_size=text_file.stat().st_size,
                    created_at=datetime.fromtimestamp(text_file.stat().st_ctime),
                    processed_at=datetime.now()
                )
                
                # 插入文档
                self.insert_document(document)
                
            except Exception as e:
                logger.error(f"Failed to process file {text_file}: {e}")
                continue
        
        logger.info(f"Directory insertion completed")
        
    except Exception as e:
        raise RAGError(f"Failed to insert from directory {directory_path}: {e}")
```

## 📈 性能优化特性

### 1. 嵌入向量优化

- **维度**：3072维（使用 `text-embedding-3-large`）
- **回退机制**：OpenAI失败时使用哈希嵌入
- **批处理**：支持批量嵌入计算

### 2. 缓存优化

- **相似度阈值**：0.90（可配置）
- **缓存大小限制**：1000个条目
- **FIFO策略**：自动清理旧缓存

### 3. 超时控制

- **LLM调用超时**：30分钟
- **查询超时**：30秒（naive模式），15秒（local模式）
- **重试机制**：最多5次重试，指数退避

## 🔧 配置参数

### 1. 核心配置

```yaml
rag:
  lightrag:
    working_dir: "./lightrag_cache"  # 工作目录
    enable_cache: true               # 启用缓存
    cache_similarity_threshold: 0.90 # 缓存相似度阈值
    openai:
      api_key: "your-api-key"       # OpenAI API密钥
```

### 2. 模型配置

```yaml
question_generator:
  local:
    model_name: "deepseek-r1:32b"   # 本地模型名称
    base_url: "http://localhost:11434" # Ollama服务地址
    temperature: 0.7                 # 温度参数
    max_tokens: 2048                 # 最大令牌数
```

## 🎯 最佳实践

### 1. 文档预处理

- **格式统一**：确保文档格式一致
- **编码处理**：使用UTF-8编码
- **内容清理**：移除无关的格式标记

### 2. 批量处理

- **分批处理**：大量文档分批插入
- **错误处理**：单个文档失败不影响整体
- **进度监控**：使用tqdm显示进度

### 3. 查询优化

- **问题规范化**：统一问题格式
- **缓存利用**：充分利用缓存机制
- **超时设置**：合理设置超时时间

### 4. 存储管理

- **目录结构**：保持工作目录整洁
- **备份策略**：定期备份知识图谱
- **清理机制**：定期清理临时文件

## 🚨 注意事项

### 1. 兼容性问题

- **tiktoken编码**：使用 `cl100k_base` 而不是 `o200k_base`
- **版本兼容**：注意LightRAG版本兼容性
- **依赖管理**：确保所有依赖正确安装

### 2. 性能考虑

- **内存使用**：大量文档可能消耗大量内存
- **处理时间**：知识图谱构建需要较长时间
- **存储空间**：嵌入向量占用大量存储空间

### 3. 错误处理

- **网络问题**：处理API调用失败
- **模型问题**：处理模型响应异常
- **存储问题**：处理文件系统错误

---

## 📚 总结

LightRAG 知识图谱生成是一个复杂的过程，涉及：

1. **文档预处理**：文本分块、格式清理
2. **实体提取**：使用LLM识别实体
3. **关系识别**：分析实体间关系
4. **图谱构建**：构建结构化知识图谱
5. **向量化存储**：生成嵌入向量
6. **查询优化**：支持多种查询模式
7. **缓存机制**：提高查询效率

通过合理的配置和优化，LightRAG 能够构建高质量的知识图谱，支持高效的语义检索和问答生成。
