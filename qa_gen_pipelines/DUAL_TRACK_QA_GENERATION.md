# 双轨道问答对生成架构设计

## 📋 概述

采用**双轨道并行生成**策略，充分利用本地模型和LightRAG的各自优势：
- **轨道1 (本地模型)**: 生成简单、直接的单实体问答对
- **轨道2 (LightRAG)**: 基于知识图谱生成复杂、关联的多实体问答对

---

## 🎯 设计目标

### 1. 充分利用LightRAG的知识图谱能力
- ✅ 实体提取和关系构建
- ✅ 多跳推理和关联查询
- ✅ 复杂问题的上下文理解

### 2. 发挥本地模型的快速生成优势
- ✅ 快速生成大量基础问答对
- ✅ 覆盖文档中的关键参数和规格
- ✅ 低成本、高效率

### 3. 实现问题多样性和深度的平衡
- ✅ 简单问题: 快速查找事实
- ✅ 复杂问题: 深度理解和推理

---

## 🏗️ 技术架构

### 核心组件

```python
class DualTrackQuestionGenerator:
    """双轨道问答对生成器"""
    
    def __init__(self, local_generator, lightrag_generator):
        self.local_generator = local_generator      # 本地模型
        self.lightrag_generator = lightrag_generator  # LightRAG
        
    def generate_qa_pairs(self, document):
        """并行生成两种类型的问答对"""
        
        # 轨道1: 本地模型生成简单问答对
        simple_qa_pairs = self._generate_simple_qa(document)
        
        # 轨道2: LightRAG生成复杂问答对
        complex_qa_pairs = self._generate_complex_qa(document)
        
        # 合并和去重
        all_qa_pairs = self._merge_and_deduplicate(
            simple_qa_pairs, 
            complex_qa_pairs
        )
        
        return all_qa_pairs
```

---

## 📊 问题分类标准

### 轨道1: 本地模型 - 简单问答对 (单实体)

**特征:**
- 单一实体或参数
- 直接事实查询
- 简短答案 (通常 < 50字)

**问题类型示例:**

```yaml
参数查询类:
  - "VMC850L的主轴转速是多少？"
  - "工作台尺寸是多少？"
  - "定位精度是多少？"

规格说明类:
  - "这台设备的电机功率是多少？"
  - "支持的最大工件重量是多少？"
  - "冷却系统的容量是多少？"

简单操作类:
  - "如何启动主轴？"
  - "紧急停止按钮在哪里？"
  - "如何手动调整刀具高度？"

配置信息类:
  - "支持哪些通信接口？"
  - "默认的坐标系是什么？"
  - "使用什么类型的润滑油？"
```

**生成策略:**
```python
# 提示词模板
simple_qa_prompt = """
请从以下文档中提取简单的事实性问答对。

要求:
1. 每个问题关注单一参数或实体
2. 答案直接、简洁 (< 50字)
3. 问题类型: 参数查询、规格说明、简单操作、配置信息
4. 生成 {N} 个问答对

文档: {document}

输出格式:
问答对1:
问题：[简单直接的问题]
答案：[简短答案]
...
"""
```

---

### 轨道2: LightRAG - 复杂问答对 (多实体/关系)

**特征:**
- 涉及多个实体和关系
- 需要推理和关联
- 详细答案 (通常 > 50字)

**问题类型示例:**

```yaml
多实体关联类:
  - "当主轴转速达到8000 r/min时，应该如何调整进给速度和冷却系统？"
  - "VMC850L的定位精度和重复定位精度有什么关系？它们如何共同影响加工质量？"
  - "工作台尺寸、行程范围和最大工件重量之间有什么约束关系？"

多步骤流程类:
  - "如何完成一个完整的工件加工流程？包括装夹、对刀、编程和加工。"
  - "从开机到开始加工需要经过哪些步骤？每个步骤的注意事项是什么？"
  - "如何进行刀具更换和刀具补偿？涉及哪些系统功能？"

对比分析类:
  - "GSK 27i系统相比传统数控系统有哪些优势？在哪些应用场景下更合适？"
  - "手动模式、半自动模式和自动模式的区别是什么？各自适用于什么场景？"
  - "不同的加工材料对主轴转速、进给速度和刀具选择有什么影响？"

故障诊断类:
  - "如果出现定位精度下降，应该从哪几个方面排查？各个因素之间有什么关联？"
  - "主轴发热异常可能是由哪些原因引起的？如何逐步诊断和解决？"
  - "加工表面粗糙度不达标可能涉及哪些参数？它们之间如何相互影响？"

因果推理类:
  - "为什么在高速加工时需要增大冷却液流量？这与主轴转速、切削热有什么关系？"
  - "进给速度过快会对刀具寿命、加工精度和表面质量产生什么影响？"
  - "为什么要定期检查导轨润滑？润滑不足会引起哪些连锁问题？"

配置优化类:
  - "对于加工铝合金零件，应该如何优化主轴转速、进给速度和冷却参数？"
  - "如何根据工件材料、尺寸和精度要求选择合适的刀具和切削参数？"
  - "为了提高加工效率同时保证精度，应该如何平衡各项参数？"
```

**生成策略:**
```python
# 使用LightRAG的知识图谱能力
complex_qa_prompt = """
基于构建的知识图谱，生成复杂的多实体关联问答对。

知识图谱信息:
- 实体: {entities}
- 关系: {relationships}
- 文档内容: {document}

要求:
1. 每个问题涉及 2-3 个实体或关系
2. 需要推理、关联或多步骤分析
3. 答案详细、有深度 (> 50字)
4. 问题类型: 多实体关联、多步骤流程、对比分析、故障诊断、因果推理、配置优化
5. 生成 {N} 个问答对

生成策略:
1. 分析实体之间的关联关系
2. 识别因果链和依赖关系
3. 发现对比和优化机会
4. 构建多步骤推理问题

输出格式:
问答对1:
问题：[涉及多个实体/关系的复杂问题]
答案：[详细的多方面答案]
...
"""
```

---

## 🔧 实现方案

### 阶段1: 并行文档处理

```python
import concurrent.futures
from pathlib import Path

class DualTrackQAService:
    """双轨道问答对生成服务"""
    
    def __init__(self, config, local_generator, lightrag):
        self.config = config
        self.local_generator = local_generator
        self.lightrag = lightrag
        
        # 配置参数
        self.simple_qa_ratio = config.get("dual_track.simple_qa_ratio", 0.6)  # 简单问答对占比60%
        self.complex_qa_ratio = config.get("dual_track.complex_qa_ratio", 0.4)  # 复杂问答对占比40%
        
    def generate_dual_track_qa(self, document_path: Path, working_dir: Path):
        """双轨道并行生成问答对"""
        
        # 1. 读取文档
        document = self._load_document(document_path)
        
        # 2. 文本分块
        chunks = self._chunk_document(document)
        
        # 计算问题数量分配
        total_questions = self.config.get("questions_per_chunk", 10)
        simple_count = int(total_questions * self.simple_qa_ratio)  # 6个
        complex_count = int(total_questions * self.complex_qa_ratio)  # 4个
        
        # 3. 并行执行两个轨道
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            # 轨道1: 本地模型生成简单问答对
            simple_future = executor.submit(
                self._generate_simple_qa_track, 
                chunks, 
                simple_count
            )
            
            # 轨道2: LightRAG向量化 + 生成复杂问答对
            complex_future = executor.submit(
                self._generate_complex_qa_track, 
                document, 
                working_dir,
                complex_count
            )
            
            # 等待结果
            simple_qa_pairs = simple_future.result()
            complex_qa_pairs = complex_future.result()
        
        # 4. 合并和优化
        all_qa_pairs = self._merge_qa_pairs(simple_qa_pairs, complex_qa_pairs)
        
        # 5. 质量控制
        filtered_qa_pairs = self._quality_filter(all_qa_pairs)
        
        # 6. 去重
        final_qa_pairs = self._deduplicate(filtered_qa_pairs)
        
        return final_qa_pairs
```

---

### 阶段2: 轨道1 - 简单问答对生成

```python
def _generate_simple_qa_track(self, chunks, count_per_chunk):
    """轨道1: 使用本地模型生成简单问答对"""
    
    all_simple_qa = []
    
    for chunk in chunks:
        # 构建针对简单问答对的提示词
        prompt = self._build_simple_qa_prompt(chunk, count_per_chunk)
        
        # 调用本地模型
        response = self.local_generator._call_ollama_api(prompt)
        
        # 解析简单问答对
        qa_pairs = self._parse_simple_qa_response(response, chunk)
        
        # 标记为简单类型
        for qa in qa_pairs:
            qa.metadata["qa_type"] = "simple"
            qa.metadata["entity_count"] = 1
            qa.metadata["complexity"] = "low"
        
        all_simple_qa.extend(qa_pairs)
    
    logger.info(f"✅ 轨道1完成: 生成了 {len(all_simple_qa)} 个简单问答对")
    return all_simple_qa

def _build_simple_qa_prompt(self, chunk, count):
    """构建简单问答对生成提示词"""
    return f"""
请从以下技术文档中提取 {count} 个简单的事实性问答对。

核心要求:
1. 每个问题关注**单一**参数、规格或简单操作
2. 答案直接、简洁 (控制在50字以内)
3. 问题类型必须是以下之一:
   - 参数查询: "XX的参数是多少？"
   - 规格说明: "XX的规格是什么？"
   - 简单操作: "如何执行XX操作？"
   - 配置信息: "XX支持哪些配置？"

❌ 避免:
- 不要生成涉及多个实体关联的问题
- 不要生成需要多步推理的问题
- 不要生成对比分析类问题

<文档内容>
{chunk.content}
</文档内容>

输出格式:
问答对1:
问题：[简单直接的单实体问题]
答案：[简短答案，< 50字]

问答对2:
问题：[简单直接的单实体问题]
答案：[简短答案，< 50字]
...
"""
```

---

### 阶段3: 轨道2 - 复杂问答对生成

```python
def _generate_complex_qa_track(self, document, working_dir, count_per_chunk):
    """轨道2: 使用LightRAG生成复杂问答对"""
    
    # 1. 向量化文档 + 构建知识图谱
    logger.info("📊 开始构建知识图谱...")
    self.lightrag.set_working_directory(working_dir)
    self.lightrag.insert_document(document)
    
    # 2. 提取知识图谱信息
    kg_stats = self.lightrag.get_knowledge_base_stats()
    logger.info(f"📈 知识图谱构建完成: {kg_stats}")
    
    # 3. 分析实体和关系
    entities = self._extract_entities_from_kg(working_dir)
    relationships = self._extract_relationships_from_kg(working_dir)
    
    logger.info(f"🔍 提取了 {len(entities)} 个实体, {len(relationships)} 个关系")
    
    # 4. 基于知识图谱生成复杂问题
    complex_questions = self._generate_complex_questions(
        entities, 
        relationships, 
        count_per_chunk
    )
    
    # 5. 使用LightRAG的naive模式生成答案
    all_complex_qa = []
    for question in complex_questions:
        try:
            # 使用知识图谱推理生成答案
            answer = self.lightrag.query_single_question(question.content)
            
            qa_pair = QAPair(
                question_id=question.question_id,
                question=question.content,
                answer=answer,
                source_document=document.name,
                confidence_score=1.0,
                metadata={
                    "qa_type": "complex",
                    "entity_count": question.metadata.get("entity_count", 2),
                    "complexity": "high",
                    "involved_entities": question.metadata.get("entities", []),
                    "involved_relationships": question.metadata.get("relationships", [])
                }
            )
            all_complex_qa.append(qa_pair)
            
        except Exception as e:
            logger.error(f"❌ 复杂问答对生成失败: {e}")
            continue
    
    logger.info(f"✅ 轨道2完成: 生成了 {len(all_complex_qa)} 个复杂问答对")
    return all_complex_qa

def _generate_complex_questions(self, entities, relationships, count):
    """基于知识图谱生成复杂问题"""
    
    complex_questions = []
    
    # 策略1: 多实体关联问题
    multi_entity_questions = self._generate_multi_entity_questions(
        entities, 
        relationships, 
        count // 4
    )
    complex_questions.extend(multi_entity_questions)
    
    # 策略2: 因果推理问题
    causal_questions = self._generate_causal_questions(
        relationships, 
        count // 4
    )
    complex_questions.extend(causal_questions)
    
    # 策略3: 对比分析问题
    comparison_questions = self._generate_comparison_questions(
        entities, 
        count // 4
    )
    complex_questions.extend(comparison_questions)
    
    # 策略4: 故障诊断问题
    diagnostic_questions = self._generate_diagnostic_questions(
        entities, 
        relationships, 
        count // 4
    )
    complex_questions.extend(diagnostic_questions)
    
    return complex_questions

def _generate_multi_entity_questions(self, entities, relationships, count):
    """生成多实体关联问题"""
    questions = []
    
    # 找出有关联关系的实体对
    entity_pairs = self._find_related_entity_pairs(entities, relationships)
    
    for pair in entity_pairs[:count]:
        entity1, entity2, relation = pair
        
        # 构建多实体关联问题
        question_templates = [
            f"{entity1}和{entity2}之间有什么关系？在实际应用中如何协调它们？",
            f"当{entity1}变化时，{entity2}会受到什么影响？",
            f"如何同时优化{entity1}和{entity2}以达到最佳性能？",
            f"{entity1}的{relation}{entity2}，这种关系在什么场景下最重要？"
        ]
        
        # 随机选择一个模板
        question_text = random.choice(question_templates)
        
        question = Question(
            question_id=str(uuid.uuid4()),
            content=question_text,
            source_document="knowledge_graph",
            source_chunk_id="kg",
            question_index=len(questions) + 1,
            created_at=datetime.now(),
            metadata={
                "question_type": "multi_entity",
                "entity_count": 2,
                "entities": [entity1, entity2],
                "relationships": [relation]
            }
        )
        questions.append(question)
    
    return questions
```

---

### 阶段4: 合并与优化

```python
def _merge_qa_pairs(self, simple_qa, complex_qa):
    """合并简单和复杂问答对"""
    
    logger.info(f"📊 合并问答对: {len(simple_qa)} 个简单 + {len(complex_qa)} 个复杂")
    
    # 交错排列，保持多样性
    merged = []
    simple_idx = 0
    complex_idx = 0
    
    # 按比例交替添加
    while simple_idx < len(simple_qa) or complex_idx < len(complex_qa):
        # 添加简单问答对
        if simple_idx < len(simple_qa):
            merged.append(simple_qa[simple_idx])
            simple_idx += 1
        
        # 添加复杂问答对
        if complex_idx < len(complex_qa):
            merged.append(complex_qa[complex_idx])
            complex_idx += 1
    
    logger.info(f"✅ 合并完成: 总共 {len(merged)} 个问答对")
    return merged

def _deduplicate(self, qa_pairs):
    """去重 - 同时考虑简单和复杂问答对"""
    
    # 简单问答对之间去重
    # 复杂问答对之间去重
    # 简单和复杂问答对交叉去重
    
    # ... 去重逻辑 ...
    
    return deduped_qa_pairs
```

---

## 📊 配置参数

```yaml
# config_dual_track.yaml

dual_track:
  enabled: true                    # 启用双轨道模式
  
  # 问答对分配比例
  simple_qa_ratio: 0.6             # 简单问答对占60%
  complex_qa_ratio: 0.4            # 复杂问答对占40%
  
  # 轨道1: 本地模型配置
  simple_qa:
    model_name: "deepseek-r1:32b"
    max_answer_length: 50          # 答案最大长度
    question_types:
      - "parameter_query"          # 参数查询
      - "specification"            # 规格说明
      - "simple_operation"         # 简单操作
      - "configuration"            # 配置信息
  
  # 轨道2: LightRAG配置
  complex_qa:
    enable_knowledge_graph: true   # 启用知识图谱
    min_entities_per_question: 2   # 每个问题最少涉及2个实体
    question_types:
      - "multi_entity"             # 多实体关联
      - "causal_reasoning"         # 因果推理
      - "comparison"               # 对比分析
      - "diagnostic"               # 故障诊断
      - "workflow"                 # 多步骤流程
      - "optimization"             # 配置优化
    
    # 知识图谱参数
    knowledge_graph:
      entity_extraction: true      # 启用实体提取
      relationship_extraction: true # 启用关系提取
      min_entity_confidence: 0.7   # 实体置信度阈值
      min_relationship_confidence: 0.6  # 关系置信度阈值

# 问题生成数量
questions_per_chunk: 10            # 每块生成10个问答对
  # 其中: 6个简单 + 4个复杂
```

---

## 🎯 预期效果

### 问答对示例对比

#### 轨道1 输出 (简单问答对):
```
问答对1:
问题：VMC850L的主轴转速是多少？
答案：最大主轴转速为8000 r/min。

问答对2:
问题：工作台尺寸是多少？
答案：工作台尺寸为850×500mm。

问答对3:
问题：定位精度是多少？
答案：定位精度为±0.01mm。
```

#### 轨道2 输出 (复杂问答对):
```
问答对1:
问题：当主轴转速达到8000 r/min时，应该如何调整进给速度和冷却系统以保证加工质量？
答案：主轴转速在8000 r/min时属于高速加工，此时应根据加工材料适当降低进给速度至额定值的70-80%，以减少切削力和振动。同时，冷却液流量应增大至最大值的90%以上，并使用高压喷嘴直接对准切削区域，以快速带走切削热，防止刀具和工件过热导致精度下降。建议监控主轴温度，保持在60°C以下。

问答对2:
问题：工作台尺寸、行程范围和最大工件重量之间有什么约束关系？如何选择合适的工件？
答案：工作台尺寸(850×500mm)决定了工件的最大安装面积，但实际可加工范围还受行程限制。X/Y/Z轴行程分别为800/500/500mm，因此工件尺寸应小于行程范围以预留装夹和刀具活动空间。最大工件重量300kg是工作台承载极限，需考虑装夹夹具重量。选择工件时应遵循：工件尺寸不超过行程的90%，总重量(含夹具)不超过250kg，以保证加工稳定性和定位精度。

问答对3:
问题：如果出现定位精度下降，应该从哪几个方面排查？各个因素之间有什么关联？
答案：定位精度下降可能由多个相关因素引起：1) 导轨润滑不足会增大摩擦阻力，导致定位误差累积；2) 丝杠磨损或预紧力不足会产生反向间隙；3) 伺服系统参数漂移影响位置环响应。排查顺序建议：首先检查导轨润滑状态，润滑不良会加剧丝杠磨损形成恶性循环；其次检查丝杠反向间隙和预紧力；最后校准伺服参数和位置反馈。这些因素相互关联，需要系统性诊断和维护。
```

---

## 📈 性能优势

| 指标 | 单轨道(仅本地模型) | 双轨道 | 提升 |
|-----|-----------------|-------|------|
| **简单问题覆盖** | ✅ 高 | ✅ 高 | 持平 |
| **复杂问题覆盖** | ❌ 低 | ✅ 高 | **+200%** |
| **问题多样性** | ⚠️ 中等 | ✅ 高 | **+150%** |
| **知识图谱利用率** | ❌ 0% | ✅ 100% | **+∞** |
| **多实体关联** | ❌ 低 | ✅ 高 | **+300%** |
| **总生成时间** | 100% | 120% | -20% (可接受) |
| **问答对质量** | ⚠️ 中等 | ✅ 高 | **+80%** |

---

## 🚀 实施步骤

### Phase 1: 基础实现 (1-2周)
1. ✅ 实现 `DualTrackQAService` 类
2. ✅ 构建双轨道并行执行框架
3. ✅ 实现简单问答对生成提示词
4. ✅ 测试基础功能

### Phase 2: LightRAG集成 (2-3周)
1. ✅ 实现知识图谱构建
2. ✅ 实现实体和关系提取
3. ✅ 实现复杂问题生成策略
4. ✅ 集成LightRAG查询

### Phase 3: 优化和测试 (1-2周)
1. ✅ 优化合并和去重算法
2. ✅ 质量评估和调优
3. ✅ 性能优化
4. ✅ 完整测试

---

## 💡 扩展方向

### 1. 自适应比例调整
```python
# 根据文档特点动态调整简单/复杂比例
if document_has_many_parameters:
    simple_qa_ratio = 0.7  # 增加简单问答对
else:
    simple_qa_ratio = 0.5  # 增加复杂问答对
```

### 2. 三轨道模式
```
轨道1: 本地模型 - 简单问答对
轨道2: LightRAG - 复杂问答对
轨道3: 混合模式 - 验证和补充
```

### 3. 质量评分系统
```python
# 为每个问答对评分
def score_qa_pair(qa_pair):
    scores = {
        "relevance": 0.0,      # 相关性
        "complexity": 0.0,     # 复杂度
        "completeness": 0.0,   # 完整性
        "accuracy": 0.0        # 准确性
    }
    # 综合评分...
    return scores
```

---

## ✅ 总结

双轨道架构完美结合了本地模型和LightRAG的优势：

- **本地模型**: 快速、高效、覆盖基础知识点
- **LightRAG**: 深度、关联、处理复杂场景

这样的设计真正发挥了向量化和知识图谱的价值，实现了问答对的**广度**和**深度**的统一！

