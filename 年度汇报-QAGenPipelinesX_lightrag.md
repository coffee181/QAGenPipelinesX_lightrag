# QAGenPipelinesX_lightrag 系统年度汇报

## 项目概述

### 项目名称
**QA Generation Pipelines X (基于 LightRAG)**

### 项目定位
智能化技术文档问答对生成系统，专注于数控机床、数控系统等工业技术领域的知识抽取与问答数据集构建

### 核心价值
- **自动化**：从PDF文档到高质量QA对的全流程自动化
- **本地化**：完全基于本地模型部署，无数据泄露风险
- **专业性**：针对工业技术领域深度优化
- **可扩展**：模块化架构，支持灵活配置与扩展

---

## 系统架构

### 1. 整体架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                     命令行交互层 (CLI)                        │
│                  main.py + scripts/*                         │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                       服务层 (Services)                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ PDF          │  │ Question     │  │ Answer       │      │
│  │ Processor    │  │ Service      │  │ Service      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│           │                │                  │              │
│  ┌────────┴────────────────┴──────────────────┴────┐       │
│  │          Progress Manager (进度跟踪)            │       │
│  └─────────────────────────────────────────────────┘       │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                    实现层 (Implementations)                   │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ OCR Engine │  │ Question   │  │ LightRAG   │            │
│  │ (Tesseract/│  │ Generator  │  │ RAG        │            │
│  │  Paddle)   │  │ (Local LLM)│  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌────────────┐  ┌────────────┐                             │
│  │ Text       │  │ Markdown   │                             │
│  │ Chunker    │  │ Processor  │                             │
│  └────────────┘  └────────────┘                             │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────┴─────────────────────────────────┐
│                    基础设施层 (Infrastructure)                │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Config     │  │ Logging    │  │ File Utils │            │
│  │ Manager    │  │ System     │  │            │            │
│  └────────────┘  └────────────┘  └────────────┘            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Path Utils │  │ Console    │  │ Chunk      │            │
│  │            │  │ Utils      │  │ Repository │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
```

### 2. 数据流程图

```
PDF文档 → OCR识别 → 文本清洗 → 智能切块 → 向量化存储
                                        ↓
                              知识图谱构建 (LightRAG)
                                        ↓
                              ┌─────────┴─────────┐
                              │                   │
                         问题生成              问题回答
                      (Knowledge Graph      (RAG检索 +
                       Context Enhanced)     引用标注)
                              │                   │
                              └─────────┬─────────┘
                                        ↓
                                   QA对数据集
                              (JSONL格式输出)
```

---

## 核心技术特性

### 1. 本地化大模型部署

#### 模型选择
- **主力模型**: DeepSeek-R1:32B (通过 Ollama 部署)
- **嵌入模型**: BGE-M3 (1024维向量)
- **服务方式**: 本地Ollama服务 (localhost:11434)

#### 技术优势
- ✅ 数据完全本地化，符合工业保密要求
- ✅ 推理速度可控，支持批处理
- ✅ 成本可控，无API调用费用

### 2. LightRAG 知识图谱增强检索

#### 核心机制
- **混合检索**: 向量检索 + 知识图谱检索
- **上下文构建**: 实体关系网络 + 文本块相似度
- **引用追溯**: 精确到chunk级别的答案来源标注

#### 配置参数
```yaml
query:
  top_k: 40                    # KG实体/关系检索数量
  chunk_top_k: 20              # 文本块保留数量
  max_total_tokens: 40000      # 总上下文token预算
  cosine_threshold: 0.2        # 向量相似度阈值
  enable_rerank: true          # 启用chunk重排序
```

### 3. 智能问题生成

#### 知识图谱上下文增强
- 每个文本块在生成问题前，先检索关联的实体和关系
- 提供最多5个实体、5个关系、2个相关文本片段作为上下文
- 确保生成的问题深度覆盖技术细节

#### 问题质量控制
- **去重机制**: 相似度阈值0.85，避免重复问题
- **质量过滤**: 自动过滤泛化、模糊问题
- **多粒度覆盖**: 基础类、中等类、深度类问题均衡生成

### 4. 进度管理与容错

#### 进度追踪
- 会话级别进度保存 (progress.json)
- 实时百分比显示
- 失败文件记录与重试机制

#### 增量保存
- 每生成5个QA对自动保存
- 支持断点续跑 (--resume)
- 信号捕获与优雅退出

### 5. OCR 文档处理

#### 双引擎支持
- **Tesseract OCR**: 中英文混合识别 (chi_sim+eng)
- **PaddleOCR**: 备选方案，中文优化

#### 优化策略
- 图像预处理（二值化、中值滤波）
- 置信度过滤（最低30%）
- 分页批处理（5页/批，降低内存峰值）

---

## 工作流程

### 完整Pipeline

#### 阶段1: PDF预处理
```bash
python scripts/preprocess_pdfs.py --input working/raw
```
- 输入: PDF文档
- 处理: OCR识别 + 文本清洗
- 输出: working/processed/*.txt

#### 阶段2: 文本向量化
```bash
python scripts/vectorize_texts.py --input working/processed
```
- 输入: 纯文本文档
- 处理: Token级切块 (1200 tokens/chunk) + 向量化 + 知识图谱构建
- 输出: working/vectorized/ (向量数据库 + 知识图谱)

#### 阶段3: QA对生成
```bash
python scripts/generate_qa_pairs.py --input working/processed --vectors working/vectorized
```
- 输入: 文本文档 + 向量知识库
- 处理: 问题生成 → 答案生成
- 输出: 
  - working/questions/*.jsonl (问题集)
  - working/qa-pairs/*.jsonl (QA对)

### 并行优化模式
```bash
python main.py generate-qapairs --input working/processed \
  --output-questions-file working/questions \
  --working-dir working/vectorized \
  --output-file working/qa-pairs \
  -i -d
```
- `-i`: 启用并行插入文档模式
- `-d`: 目录批处理模式
- 同时执行问题生成和文档插入，提升40%效率

---

## 系统能力指标

### 处理能力
- **文档处理**: 支持220+ PDF文档
- **QA对生成**: 已生成34个领域的问答数据集
- **并发能力**: 支持问题生成和文档插入并行执行

### 质量指标
- **问题覆盖**: 每chunk生成10个问题，覆盖基础/中等/深度三个粒度
- **答案准确性**: 基于RAG检索，包含精确chunk引用
- **去重效果**: 相似度阈值0.85，有效避免重复

### 可靠性
- **进度持久化**: 每处理1个文件保存一次
- **错误恢复**: 支持断点续跑
- **优雅退出**: 信号捕获 + 自动保存

---

## 技术栈

### 核心依赖
| 组件 | 技术 | 版本 | 用途 |
|------|------|------|------|
| RAG框架 | LightRAG | 1.3.0+ | 知识图谱增强检索 |
| 大模型 | DeepSeek-R1 | 32B | 问题生成与答案生成 |
| 嵌入模型 | BGE-M3 | 1024维 | 文本向量化 |
| OCR引擎 | Tesseract / PaddleOCR | - | PDF文本识别 |
| 配置管理 | PyYAML | 6.0+ | 系统配置 |
| 进度管理 | JSONL + Custom | - | 会话追踪 |
| 文本处理 | NLTK + Tiktoken | - | 分句与token计数 |

### 开发框架
- **语言**: Python 3.11+
- **接口设计**: 基于抽象接口的分层架构
- **日志**: Loguru + UTF-8编码
- **CLI**: argparse多命令模式

---

## 项目成果

### 数据集构建
- ✅ 处理220+ 数控系统技术文档
- ✅ 生成34个细分领域的QA数据集
- ✅ 覆盖GSK系列数控系统、机床设备、驱动系统等

### 数据集示例
```
working/qa-pairs/
├── 01-GSK 27i高端多通道数控系统 20250226_qa_pairs.jsonl
├── 01-宝鸡机床VMC850L_qa_pairs.jsonl
├── 02-GSK 25i系列五轴加工中心数控系统 20250317_qa_pairs.jsonl
├── 03-GSK 25Ti车铣复合数控系统 20250307_qa_pairs.jsonl
└── ... (共34个文件)
```

### 质量样本
**问题**: "GSK988TA在主轴8000 r/min运行时，外喷与中心冷却应如何切换以维持工件温度？"

**答案**: "根据文档，GSK988TA主轴系统最高转速8000 r/min，配备外喷和中心冷却系统..."

**引用**: `[chunk_id: abc123, source: GSK988TA手册 第4页]`

---

## 配置灵活性

### 多层次配置
```yaml
# OCR配置
ocr.provider: tesseract / paddle
ocr.tesseract.lang: chi_sim+eng
ocr.tesseract.min_confidence: 30

# 文本切块
text_chunker.use_token_chunking: true
text_chunker.chunk_token_size: 1200
text_chunker.chunk_overlap_token_size: 100
text_chunker.persist_chunks: true

# 问题生成
question_generator.provider: local
question_generator.local.model_name: deepseek-r1:32b
question_generator.local.questions_per_chunk: 10
question_generator.enable_kg_context: true

# RAG检索
rag.lightrag.query.top_k: 40
rag.lightrag.query.chunk_top_k: 20
rag.lightrag.query.enable_rerank: true
```

### 环境适配
- ✅ Windows/Linux跨平台
- ✅ 中文路径完美支持
- ✅ 控制台UTF-8自动修复
- ✅ PyInstaller打包支持

---

## 未来规划

### 短期优化 (Q1-Q2)
1. **答案质量评估**
   - 引入自动评分机制
   - 人工标注少量样本作为基准

2. **多模态支持**
   - 表格识别与解析
   - 图表描述生成

3. **性能提升**
   - GPU加速推理
   - 并行度进一步提升

### 中期拓展 (Q3-Q4)
1. **领域扩展**
   - 其他工业领域文档
   - 多语言支持（英文、日文）

2. **交互式界面**
   - Web UI for QA对标注与审核
   - 实时进度监控面板

3. **模型微调**
   - 基于生成的QA对微调领域模型
   - 提升问题生成的专业性

### 长期愿景
- 构建工业技术领域最大的中文QA数据集
- 支撑领域专用大模型的训练与评测
- 形成可复用的文档知识抽取解决方案

---

## 技术亮点

### 1. 知识图谱增强的问题生成
传统方法只看单个文本块，本系统通过LightRAG提供实体关系上下文，生成的问题更具深度和关联性。

### 2. Token级精确切块
与LightRAG保持一致的Token切块策略（1200 tokens/chunk），避免向量检索时的语义错位。

### 3. 双引擎容错
Tesseract与PaddleOCR互为备份，确保OCR环节的高可用性。

### 4. 增量式QA生成
自动保存机制 + 断点续跑，长时间任务安全可控。

### 5. 完善的进度追踪
会话级进度管理、百分比里程碑、失败文件记录，运维友好。

---

## 项目总结

### 核心成就
✅ **技术创新**: 知识图谱增强 + 本地大模型 + 精确引用追溯  
✅ **工程完善**: 模块化架构 + 容错机制 + 进度管理  
✅ **数据成果**: 220+ 文档处理 + 34个QA数据集  
✅ **可扩展性**: 接口化设计 + 灵活配置 + 跨平台支持  

### 技术价值
- 为工业技术领域提供了可落地的知识抽取方案
- 验证了本地化大模型在垂直领域的可行性
- 构建了从文档到QA对的完整自动化流水线

### 业务价值
- 大幅降低技术文档问答数据集构建成本
- 支撑智能客服、技术咨询等下游应用
- 为领域模型训练提供高质量数据

---

## 附录

### 目录结构
```
QAGenPipelinesX_lightrag/
├── qa_gen_pipelines/         # 主程序目录
│   ├── main.py              # CLI入口
│   ├── config_local.yaml    # 配置文件
│   ├── src/                 # 源代码
│   │   ├── implementations/ # 实现层
│   │   ├── interfaces/      # 接口层
│   │   ├── models/          # 数据模型
│   │   ├── services/        # 服务层
│   │   └── utils/           # 工具类
│   ├── scripts/             # 脚本工具
│   │   ├── preprocess_pdfs.py
│   │   ├── vectorize_texts.py
│   │   └── generate_qa_pairs.py
│   └── requirements.txt     # 依赖列表
├── working/                  # 工作目录
│   ├── raw/                 # 原始PDF (220+)
│   ├── processed/           # 处理后文本 (220+)
│   ├── vectorized/          # 向量库 + 知识图谱
│   ├── questions/           # 问题集 (34个)
│   └── qa-pairs/            # QA对数据集 (34个)
└── README.md
```

### 关键命令速查
```bash
# 1. PDF预处理
python scripts/preprocess_pdfs.py --input working/raw

# 2. 文本向量化
python scripts/vectorize_texts.py --input working/processed

# 3. QA对生成
python scripts/generate_qa_pairs.py \
  --input working/processed \
  --vectors working/vectorized

# 4. 查看进度
python main.py show-progress --detailed

# 5. 实时监控
python main.py show-progress --monitor
```

### 性能参考
- **PDF处理**: ~30秒/文档 (平均20页)
- **向量化**: ~3分钟/文档 (包含知识图谱构建)
- **QA生成**: ~5分钟/文档 (每chunk 10个问题)
- **总计**: ~8-10分钟/文档 (端到端)

---

**报告生成时间**: 2025年12月10日  
**项目状态**: 生产就绪  
**维护团队**: AI系统开发组  
**联系方式**: 见项目README

