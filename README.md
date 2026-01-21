# QAGen Pipeline

基于 LightRAG 的工业文档问题生成流水线。从 PDF 文档中提取文本，构建知识图谱，并自动生成高质量的运维诊断问题。

## 项目结构

```
QAGen_Project/
├── config/
│   ├── config.yaml          # 全局配置 (路径、模型参数)
│   └── settings.py          # Pydantic 配置加载器
├── src/                     # 核心功能模块
│   ├── ocr_engine.py        # 封装 PaddleOCR
│   ├── rag_core.py          # 封装 LightRAG (建图、KV存储管理)
│   └── llm_client.py        # 封装 Ollama/DeepSeek 交互与 Prompt
├── steps/                   # 三个独立执行脚本
│   ├── 1_pdf_to_text.py     # 步骤1: PDF -> OCR -> working/processed
│   ├── 2_build_graph.py     # 步骤2: 文本 -> LightRAG 建图 -> working/lightrag_db
│   └── 3_gen_questions.py   # 步骤3: 遍历Chunk -> 生成问题 -> working/output
├── working/                 # 数据存储目录 (按 domain 组织)
│   ├── raw/                 # PDF输入
│   ├── processed/           # OCR后的TXT
│   ├── lightrag_db/         # LightRAG 的 graph/kv_store/vdb
│   └── output/              # 最终生成的 questions.jsonl
├── main.py                  # 调度脚本，依次调用三个步骤
└── requirements.txt         # Python 依赖
```

## 安装

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

## 前置条件

1. **Ollama 服务**: 用于本地 LLM 和 Embedding
   ```bash
   # 安装模型
   ollama pull bge-m3
   ollama pull deepseek-r1:32b
   ```

2. **PaddleOCR**: 用于 PDF 文本提取
   ```bash
   pip install paddleocr paddlex
   ```

## 使用方法

### 按行业/领域 (Domain) 组织数据

本项目支持按 `--domain` 参数组织不同行业/领域的数据，**每个 domain 拥有独立的知识图谱和问题文件**：

```
working/
├── raw/
│   ├── Robot/                    # domain: Robot
│   │   └── *.pdf
│   └── Numerical-Control-System/ # domain: Numerical-Control-System
│       └── *.pdf
├── processed/
│   ├── Robot/
│   │   └── *.txt, *.md
│   └── Numerical-Control-System/
│       └── *.txt, *.md
├── lightrag_db/
│   ├── Robot/                    # Robot 独立的知识图谱
│   │   └── (LightRAG files)
│   └── Numerical-Control-System/ # 数控系统独立的知识图谱
│       └── (LightRAG files)
└── output/
    ├── Robot/
    │   └── *_questions.jsonl
    └── Numerical-Control-System/
        └── *_questions.jsonl
```

### 方式一：指定 Domain 运行流水线

```bash
# 处理 Robot 领域
python main.py --domain Robot

# 处理数控系统领域
python main.py --domain Numerical-Control-System

# 列出可用的 domains
python main.py --list-domains
```

### 方式二：分步执行（指定 Domain）

```bash
# 步骤1: PDF 转文本 (OCR)
python steps/1_pdf_to_text.py --domain Robot

# 步骤2: 构建知识图谱
python steps/2_build_graph.py --domain Robot

# 步骤3: 生成问题
python steps/3_gen_questions.py --domain Robot
```

### 命令行参数

```bash
# 查看帮助
python main.py --help
python steps/1_pdf_to_text.py --help

# 指定配置文件
python main.py --config my_config.yaml --domain Robot

# 仅运行特定步骤
python main.py --step 1 --domain Robot        # 仅OCR
python main.py --step 2 --step 3 --domain Robot  # 仅建图和问题生成

# 列出各步骤可用的 domains
python steps/1_pdf_to_text.py --list-domains
python steps/2_build_graph.py --list-domains
python steps/3_gen_questions.py --list-domains

# 手动指定输入输出路径（覆盖默认）
python steps/1_pdf_to_text.py --input /path/to/pdfs --output /path/to/output
```

## 配置说明

编辑 `config/config.yaml` 修改配置：

```yaml
# 路径配置
paths:
  working_dir: "./working"
  raw_dir: "raw"              # PDF输入目录
  processed_dir: "processed"  # OCR输出目录
  lightrag_db_dir: "lightrag_db"  # LightRAG存储
  output_dir: "output"        # 问题输出目录

# Ollama 服务地址
lightrag:
  embedding:
    base_url: "http://localhost:11434"
    model: "bge-m3"
  llm:
    base_url: "http://localhost:11434"
    model: "deepseek-r1:32b"

# 问题生成参数
question_gen:
  questions_per_chunk: 10  # 每个chunk生成的问题数
```

## 输出格式

生成的问题保存为 JSONL 格式：

```json
{
  "question_id": "uuid",
  "content": "问题内容",
  "source_document": "文档名",
  "source_chunk_id": "chunk-xxx",
  "metadata": {
    "lightrag_chunk_id": "chunk-xxx",
    "related_entities": ["实体1", "实体2"]
  }
}
```

## 目录说明

| 目录 | 说明 |
|------|------|
| `working/raw/[domain]/` | 放入待处理的 PDF 文件 |
| `working/processed/[domain]/` | OCR 提取的文本文件 (.txt, .md) |
| `working/lightrag_db/[domain]/` | LightRAG 知识图谱存储 |
| `working/output/[domain]/` | 生成的问题文件 (_questions.jsonl) |

## 典型工作流程

```bash
# 1. 创建 domain 目录并放入 PDF
mkdir -p working/raw/MyIndustry
cp *.pdf working/raw/MyIndustry/

# 2. 运行完整流水线
python main.py --domain MyIndustry

# 3. 查看生成的问题
ls working/output/MyIndustry/
```

## License

MIT
