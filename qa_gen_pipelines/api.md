# QA生成管道 API 文档

## 概述

QA生成管道是一个完整的问答对生成系统，支持PDF处理、问题生成、答案生成和知识库管理。系统基于LightRAG实现检索增强生成(RAG)，使用DeepSeek LLM进行问题和答案生成。

## 系统架构

```
PDF文档 → PDF处理 → 问题生成 → 答案生成 → QA对输出
                ↓
            LightRAG知识库
```

## 环境配置

### 必需的环境变量
在`.env`文件中配置以下API密钥：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # 可选，用于嵌入向量
```

### 依赖安装
```bash
pip install -r requirements.txt
```

## 命令行接口

### 基本语法
```bash
python main.py [全局选项] <命令> [命令参数]
```

### 全局选项
- `--session-id SESSION_ID`: 指定会话ID用于进度跟踪
- `--help`: 显示帮助信息

## 命令详解

### 1. PDF处理 (process-pdfs)

将PDF文件处理为文本块，为后续步骤准备数据。

#### 语法
```bash
python main.py --session-id <session_id> process-pdfs <input_path> <output_dir>
```

#### 参数
- `input_path`: PDF文件或包含PDF文件的目录路径
- `output_dir`: 输出目录，处理后的文本将保存在此目录

#### 示例
```bash
# 处理单个PDF文件
python main.py --session-id pdf_process process-pdfs document.pdf ./output

# 处理目录中的所有PDF文件
python main.py --session-id batch_pdf process-pdfs ./pdf_folder ./batch_output
```

#### 输出
- 在`output_dir/texts/`目录下生成`.txt`文件
- 每个文本块包含OCR提取的内容
- 生成进度报告和统计信息

---

### 2. 问题生成 (generate-questions)

基于处理后的文本内容生成问题。

#### 语法
```bash
python main.py --session-id <session_id> generate-questions <texts_dir> <output_file>
```

#### 参数
- `texts_dir`: 包含文本文件的目录路径
- `output_file`: 输出JSONL文件路径

#### 示例
```bash
python main.py --session-id question_gen generate-questions ./output/texts questions.jsonl
```

#### 输出格式
生成的JSONL文件中每行包含一个问题对象：
```json
{
  "question_id": "q_001",
  "text": "如何正确安装和调试GSK数控系统？",
  "question_type": "technical", 
  "difficulty": "medium",
  "category": "installation",
  "tags": ["GSK", "数控系统", "安装"],
  "source": "document_name.txt"
}
```

---

### 3. 答案生成 (generate-answers)

基于问题和知识库生成答案。支持两种模式：

#### 模式1：使用已有知识库（默认模式）

使用现有的LightRAG知识库快速生成答案。

##### 语法
```bash
python main.py --session-id <session_id> generate-answers <questions_file> <working_dir> <output_file>
```

##### 参数
- `questions_file`: 包含问题的JSONL文件
- `working_dir`: LightRAG工作目录路径
- `output_file`: 输出JSONL文件路径

##### 示例
```bash
python main.py --session-id qa_existing generate-answers questions.jsonl ./my_knowledge_base output.jsonl
```

##### 特点
- ⚡ **极快速度**：利用已有知识库缓存，几乎瞬间完成
- 💾 **节省资源**：不重新处理文档
- 🔄 **可重复使用**：同一知识库可用于多批问题

#### 模式2：插入模式（-i参数）

在生成答案的同时向知识库插入新文档。

##### 语法
```bash
python main.py --session-id <session_id> generate-answers <questions_file> <working_dir> <output_file> -i <documents_path>
```

##### 参数
- `questions_file`: 包含问题的JSONL文件
- `working_dir`: LightRAG工作目录路径（如不存在则创建）
- `output_file`: 输出JSONL文件路径
- `-i <documents_path>`: 要插入的文档路径（文件或目录）

##### 示例
```bash
# 插入单个文件
python main.py --session-id qa_insert generate-answers questions.jsonl ./new_kb output.jsonl -i document.txt

# 插入目录中的所有文件
python main.py --session-id qa_batch generate-answers questions.jsonl ./kb output.jsonl -i ./texts_dir
```

##### 特点
- 🔄 **智能追加**：向已有知识库追加新文档，而非重建
- 📈 **增量更新**：保留原有知识，添加新内容
- 🏗️ **自动创建**：工作目录不存在时自动创建

#### 答案输出格式
```json
{
  "messages": [
    {
      "role": "user",
      "content": "如何正确安装和调试GSK数控系统？"
    },
    {
      "role": "assistant", 
      "content": "GSK数控系统安装与调试指南\n...\nReferences:\n[KG] 相关实体 (unknown_source)\n..."
    }
  ]
}
```

---

### 4. 文档插入 (insert-documents)

专门用于向知识库插入文档的命令。

#### 语法
```bash
python main.py --session-id <session_id> insert-documents <working_dir> <documents_path>
```

#### 参数
- `working_dir`: LightRAG工作目录路径
- `documents_path`: 要插入的文档路径（文件或目录）

#### 示例
```bash
# 插入单个文档
python main.py --session-id doc_insert insert-documents ./my_kb document.txt

# 插入目录中的所有文档
python main.py --session-id batch_insert insert-documents ./my_kb ./texts_folder
```

#### 特点
- 📝 **专用插入**：专门用于知识库管理
- 🔄 **增量添加**：向现有知识库追加内容
- 📊 **详细统计**：提供插入成功/失败的详细报告

---

## 完整工作流程示例

### 场景1：从PDF到QA对的完整流程

```bash
# 步骤1：处理PDF文件
python main.py --session-id step1 process-pdfs ./pdfs ./processed

# 步骤2：生成问题
python main.py --session-id step2 generate-questions ./processed/texts questions.jsonl

# 步骤3：生成答案（插入模式，创建知识库）
python main.py --session-id step3 generate-answers questions.jsonl ./knowledge_base qa_output.jsonl -i ./processed/texts
```

### 场景2：使用已有知识库快速生成答案

```bash
# 直接使用现有知识库
python main.py --session-id quick_qa generate-answers new_questions.jsonl ./existing_kb new_answers.jsonl
```

### 场景3：向现有知识库添加新文档

```bash
# 添加新文档到知识库
python main.py --session-id add_docs insert-documents ./my_kb new_document.txt

# 使用更新后的知识库生成答案
python main.py --session-id updated_qa generate-answers questions.jsonl ./my_kb answers.jsonl
```

---

## 进度跟踪

系统自动跟踪每个会话的进度，保存在`progress.json`文件中。

### 查看进度
可以通过日志输出查看实时进度，包括：
- 处理的文件数量
- 成功/失败统计
- 知识库统计信息
- 生成的QA对数量

---

## 性能参考

| 操作 | 处理时间 | 说明 |
|------|----------|------|
| PDF处理 | 30-60秒/文档 | 取决于文档大小和复杂度 |
| 问题生成 | 10-30秒/文本块 | 使用DeepSeek LLM |
| 答案生成（已有KB） | <1秒 | 利用知识库缓存 |
| 答案生成（插入模式） | 1-2分钟/文档 | 包含知识库构建时间 |
| 文档插入 | 30-60秒/文档 | LightRAG处理时间 |

---

## 错误处理

### 常见错误及解决方案

1. **API密钥错误**
   ```
   Error: DeepSeek API key not configured
   ```
   解决：检查`.env`文件中的`DEEPSEEK_API_KEY`配置

2. **知识库不存在**
   ```
   Error: Knowledge base directory does not exist
   ```
   解决：使用插入模式(-i)或先创建知识库

3. **NumPy版本冲突**
   ```
   Error: NumPy 2.x compatibility issue
   ```
   解决：`pip install "numpy<2"`

4. **文件编码问题**
   ```
   Error: No questions found in file
   ```
   解决：确保JSONL文件使用UTF-8编码

---

## 最佳实践

### 1. 工作目录管理
- 为不同项目使用独立的工作目录
- 定期备份重要的知识库
- 使用描述性的目录名称

### 2. 会话ID规范
- 使用有意义的会话ID便于跟踪
- 格式建议：`项目名_操作类型_日期`
- 例：`manual_qa_gen_20241225`

### 3. 性能优化
- 优先使用已有知识库（默认模式）
- 批量处理多个问题
- 定期清理不需要的工作目录

### 4. 文件组织
```
project/
├── pdfs/           # 原始PDF文件
├── processed/      # 处理后的文本
├── knowledge_bases/ # LightRAG工作目录
├── questions/      # 问题文件
├── answers/        # 答案文件
└── logs/          # 日志文件
```

---

## 技术栈

- **PDF处理**: Tesseract OCR
- **文本分块**: 自定义分块器
- **问题生成**: DeepSeek LLM
- **RAG系统**: LightRAG
- **答案生成**: LightRAG + DeepSeek LLM
- **向量嵌入**: OpenAI Embeddings (可选) 或 哈希嵌入

---

## 更新日志

### v1.0.0
- ✅ 基础PDF处理功能
- ✅ 问题生成功能
- ✅ 答案生成功能
- ✅ LightRAG集成
- ✅ 进度跟踪系统
- ✅ 知识库追加功能
- ✅ 插入模式支持
- ✅ 错误处理优化

---

## 支持与反馈

如遇到问题或需要功能改进，请提供：
1. 完整的错误信息
2. 使用的命令
3. 输入文件示例
4. 系统环境信息 