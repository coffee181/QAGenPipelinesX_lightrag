# Pipeline Scripts Usage Guide

本文档介绍 `scripts/` 目录下三个脚本的用途、常用参数以及示例命令，
帮助你快速完成 “PDF 预处理 → 文本向量化 → 问答对生成” 的流水线。

## 1. 预处理 PDF：`preprocess_pdfs.py`
将 `working/raw` 目录中的 PDF 转为纯文本 (`working/processed/*.txt`)。

### 常用参数
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input` | 指定单个 PDF 或包含 PDF 的目录 | `working/raw` |
| `--output` | 预处理文本输出目录 | `working/processed` |
| `--paths` | 显式列出若干 PDF，优先级高于 `--input` | 无 |
| `--config` | 配置文件路径 | `config.yaml` |
| `--session-id` | 进度跟踪 Session ID | 自动生成 |

### 示例
```bash
# 预处理整个 raw 目录（working 位于仓库根目录）
python scripts/preprocess_pdfs.py --input ../working/raw

# 仅处理两个指定文件
python scripts/preprocess_pdfs.py --paths \
  "../working/raw/01系统综合-铣削.pdf" \
  "../working/raw/02系统综合-车削.pdf"

# 输出至自定义目录
python scripts/preprocess_pdfs.py --input ../working/raw --output ../working/custom_processed
```

## 2. 向量化文本：`vectorize_texts.py`
读取 `working/processed` 中的 `.txt`，并通过 LightRAG 插入到向量知识库。
默认会把数据写入 `working/vectorized`，供后续问答复用。

### 常用参数
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input` | 单个 `.txt` 或目录 | `working/processed` |
| `--output` | LightRAG 工作目录 | `working/vectorized` |
| `--paths` | 显式列出若干 `.txt` 文件 | 无 |
| `--session-id` | 进度跟踪 ID | 自动生成 |

### 示例
```bash
# 将整个 processed 目录插入知识库
python scripts/vectorize_texts.py --input ../working/processed

# 插入单个文档，并指定向量目录
python scripts/vectorize_texts.py \
  --input ../working/processed/01系统综合-铣削.txt \
  --output ../working/vectorized

# 指定多文件批量插入
python scripts/vectorize_texts.py --paths \
  ../working/processed/01系统综合-铣削.txt \
  ../working/processed/02系统综合-车削.txt
```

## 3. 生成问答对：`generate_qa_pairs.py`
对已处理文本生成问题，再基于既有向量库生成答案，输出 JSONL 格式的问答对。
工作流程：1.生成问题→working/questions 2.生成答案→working/qa-pairs

### 常用参数
| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--input` | 文本文件或目录 | `working/processed` |
| `--vectors` | LightRAG 工作目录 | `working/vectorized` |
| `--output` | QA 对输出目录 | `working/qa-pairs` |
| `--questions-output` | 问题文件目录 | `working/questions` |
| `--paths` | 显式指定 `.txt` 文件列表 | 无 |
| `--resume` | 若目标 QA 文件存在，是否增量续写 | `False` |
| `--session-id` | 基础 Session ID，会附加文档名 | 自动生成 |

### 示例
```bash
# 针对 processed 目录全部文档生成问答（问题→working/questions，答案→working/qa-pairs）
python scripts/generate_qa_pairs.py --input ../working/processed --vectors ../working/vectorized --output ../working/qa-pairs

# 仅对单个文档生成问答
python scripts/generate_qa_pairs.py --input ../working/processed/01系统综合-铣削.txt --vectors ../working/vectorized --output ../working/qa-pairs --session-id qa_session_01

# 多文件显式处理，断点续跑
python scripts/generate_qa_pairs.py --paths ../working/processed/01系统综合-铣削.txt --paths ../working/processed/02系统综合-车削.txt --vectors ../working/vectorized --output ../working/qa-pairs --resume

# 自定义问题输出目录
python scripts/generate_qa_pairs.py --input ../working/processed --vectors ../working/vectorized --output ../working/qa-pairs --questions-output ../working/custom_questions
```

## 推荐流水线
```bash
python scripts/preprocess_pdfs.py --input ../working/raw
python scripts/vectorize_texts.py --input ../working/processed
python scripts/generate_qa_pairs.py --input ../working/processed --vectors ../working/vectorized
```

以上示例均可根据需要替换配置文件、会话 ID 与输出目录，满足单文档、多文档乃至全量处理的需求。
