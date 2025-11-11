# QA Generation Pipelines

A comprehensive Python application for generating Question-Answer pairs from PDF documents using OCR, LLM-based question generation, and RAG-based answer generation. Built following SOLID principles for maintainability and extensibility.

## Features

- **PDF Processing**: Convert PDFs to text using OCR (Tesseract)
- **Text Chunking**: Intelligent text segmentation for LLM processing
- **Question Generation**: Generate questions using DeepSeek LLM with GSK equipment maintenance prompts
- **Answer Generation**: Generate answers using LightRAG for retrieval-augmented generation
- **Progress Management**: Resumable operations with session tracking
- **Batch Processing**: Process multiple files and directories
- **Markdown Processing**: Clean and format LLM outputs
- **SOLID Architecture**: Extensible design with abstract interfaces

## Project Structure

```
qa_gen_pipelines/
├── src/
│   ├── interfaces/           # Abstract interfaces (SOLID principles)
│   │   ├── ocr_interface.py
│   │   ├── text_chunker_interface.py
│   │   ├── question_generator_interface.py
│   │   ├── rag_interface.py
│   │   └── markdown_processor_interface.py
│   ├── implementations/      # Concrete implementations
│   │   ├── tesseract_ocr.py
│   │   ├── simple_text_chunker.py
│   │   ├── deepseek_question_generator.py
│   │   ├── lightrag_rag.py
│   │   └── simple_markdown_processor.py
│   ├── services/            # Business logic services
│   │   ├── progress_manager.py
│   │   ├── pdf_processor.py
│   │   ├── question_service.py
│   │   └── answer_service.py
│   ├── models/              # Data models
│   │   ├── document.py
│   │   ├── question.py
│   │   └── qa_pair.py
│   └── utils/               # Utilities
│       ├── config_manager.py
│       └── file_utils.py
├── main.py                  # Main application
├── config.yaml             # Configuration file
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## 📚 文档索引

### 📖 参数文档
- **[完整参数说明](./Docs/PARAMETERS_REFERENCE.md)** - 所有参数的详细说明
- **[快速参考指南](./Docs/QUICK_REFERENCE.md)** - 常用命令和参数组合
- **[参数索引](./Docs/PARAMETER_INDEX.md)** - 按字母顺序的参数索引

### 🔧 技术文档
- **[TikToken兼容性解决方案](./Docs/TIKTOKEN_FIX.md)** - 构建问题解决指南
- **[增量保存机制](./INCREMENTAL_SAVE_README.md)** - 进度管理和恢复机制
- **[实现总结](./IMPLEMENTATION_SUMMARY.md)** - 技术实现概述

### 🚀 快速开始

如果你只想快速使用可执行文件，请查看：
- **[快速参考指南](./Docs/QUICK_REFERENCE.md)** - 最快上手方式
- **[参数索引](./Docs/PARAMETER_INDEX.md)** - 快速查找特定参数

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Tesseract OCR** (仅在需要PDF处理时):
   - Windows: Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - Linux: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`

### Setup

1. **Clone or create the project directory**:
   ```bash
   mkdir qa_gen_pipelines
   cd qa_gen_pipelines
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the application**:
   - Copy `config.yaml` and update the settings
   - Set your DeepSeek API key in the config or environment variable
   - Update Tesseract path if needed

## Configuration

Edit `config.yaml` to configure the application:

```yaml
# OCR Configuration
ocr:
  tesseract_cmd: "tesseract"  # Path to tesseract executable

# Text Chunking Configuration
chunking:
  chunk_size: 1000
  overlap_size: 200

# DeepSeek LLM Configuration
deepseek:
  api_key: "${DEEPSEEK_API_KEY}"  # Set via environment variable
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
  max_tokens: 2000
  temperature: 0.7

# RAG Configuration
rag:
  working_dir: "rag_storage"

# Progress Management
progress:
  storage_path: "progress"

# Question Generation Prompts (GSK Equipment Maintenance)
prompts:
  system_prompt: |
    You are an expert in GSK (GlaxoSmithKline) pharmaceutical equipment maintenance...
  # ... (full prompts as configured)
```

### Environment Variables

Set your DeepSeek API key:
```bash
export DEEPSEEK_API_KEY="your_api_key_here"
```

## 🚀 使用方式

### 可执行文件版本（推荐）

如果你已经有构建好的可执行文件，请参考：
- **[快速参考指南](./Docs/QUICK_REFERENCE.md)** - 最简单的使用方式
- **[完整参数说明](./Docs/PARAMETERS_REFERENCE.md)** - 详细的参数文档

基本用法：
```bash
# 设置API密钥
echo "DEEPSEEK_API_KEY=your_key_here" > .env

# 生成答案（最常用）
./qa_gen_pipeline.exe generate-answers questions.jsonl ./kb_dir/ output.jsonl

# 查看所有可用命令
./qa_gen_pipeline.exe --help
```

### Python脚本版本

The application provides several commands for different stages of the pipeline:

### 1. Process PDFs (OCR)

Convert PDF files to text:

```bash
# Single PDF
python main.py process-pdfs input.pdf output_directory/

# Directory of PDFs
python main.py process-pdfs pdf_directory/ output_directory/
```

### 2. Generate Questions

Generate questions from text documents:

```bash
# Single document
python main.py generate-questions document.txt output_directory/

# Directory of documents
python main.py generate-questions text_directory/ output_directory/
```

### 3. Generate Answers

Generate answers using RAG:

```bash
# Single questions file
python main.py generate-answers questions.jsonl documents_directory/ output_directory/

# Directory of question files
python main.py generate-answers questions_directory/ documents_directory/ output_directory/
```

### 4. Full Pipeline

Run the complete pipeline from PDFs to QA pairs:

```bash
# Single PDF
python main.py full-pipeline input.pdf output_directory/

# Directory of PDFs
python main.py full-pipeline pdf_directory/ output_directory/
```

This creates the following structure:
```
output_directory/
├── texts/          # Processed text files
├── questions/      # Generated questions
└── qa/            # Generated QA pairs
```

### 5. Progress Tracking

Monitor progress of operations:

```bash
# Show all sessions
python main.py show-progress

# Show specific session
python main.py show-progress --session-id session_123
```

### Command Line Options

- `--config`: Specify configuration file (default: `config.yaml`)
- `--log-level`: Set logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
- `--session-id`: Specify session ID for progress tracking

## File Formats

### Questions Format (`questions.jsonl`)
```json
{"messages": ["What is the maintenance procedure for...", "How often should..."]}
```

### QA Format (`qa.jsonl`)
```json
{
  "messages": [
    {"role": "user", "content": "What is the maintenance procedure for..."},
    {"role": "assistant", "content": "The maintenance procedure involves..."}
  ]
}
```

## Examples

### Example 1: Process a Single PDF
```bash
python main.py process-pdfs manual.pdf ./output/
python main.py generate-questions ./output/manual.txt ./questions/
python main.py generate-answers ./questions/manual_questions.jsonl ./output/ ./qa/
```

### Example 2: Full Pipeline for Directory
```bash
python main.py full-pipeline ./pdf_manuals/ ./complete_output/
```

### Example 3: Resume Failed Operation
```bash
# Check progress
python main.py show-progress

# Resume if needed (implementation depends on specific service)
```

## Architecture

The application follows SOLID principles:

- **Single Responsibility**: Each class has one clear purpose
- **Open/Closed**: Easy to add new implementations without modifying existing code
- **Liskov Substitution**: Implementations can be swapped seamlessly
- **Interface Segregation**: Focused, minimal interfaces
- **Dependency Inversion**: Services depend on abstractions, not concretions

### Key Components

1. **Interfaces**: Define contracts for OCR, chunking, question generation, RAG, and markdown processing
2. **Implementations**: Concrete implementations using Tesseract, DeepSeek, LightRAG, etc.
3. **Services**: High-level business logic for PDF processing, question generation, and answer generation
4. **Models**: Data structures for documents, questions, and QA pairs
5. **Progress Manager**: Session-based progress tracking with resumability

## Extending the System

### Adding New OCR Implementation

1. Create a new class implementing `OCRInterface`
2. Register it in the service factory
3. Update configuration as needed

```python
from src.interfaces.ocr_interface import OCRInterface

class MyOCRImplementation(OCRInterface):
    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        # Your implementation
        pass
```

### Adding New Question Generator

1. Implement `QuestionGeneratorInterface`
2. Add configuration options
3. Update service creation

```python
from src.interfaces.question_generator_interface import QuestionGeneratorInterface

class MyQuestionGenerator(QuestionGeneratorInterface):
    def generate_questions(self, text: str) -> List[str]:
        # Your implementation
        pass
```

## Troubleshooting

### Common Issues

1. **Tesseract not found**: Ensure Tesseract is installed and path is correct in config
2. **DeepSeek API errors**: Check API key and network connectivity
3. **Memory issues**: Reduce chunk size or process files individually
4. **Permission errors**: Ensure write permissions for output directories

### Logging

The application logs to both console and `qa_pipeline.log`. Use `--log-level DEBUG` for detailed information.

### Performance Tips

1. **Batch Processing**: Use directory commands for multiple files
2. **Chunk Size**: Adjust chunk size based on available memory
3. **Parallel Processing**: Consider implementing parallel processing for large datasets
4. **Resume Capability**: Use session IDs to resume failed operations

## Dependencies

Key dependencies include:
- `pytesseract`: OCR functionality
- `lightrag`: RAG implementation
- `openai`: LLM API client (compatible with DeepSeek)
- `PyPDF2`: PDF processing
- `pyyaml`: Configuration management
- `pathlib`: Path handling

See `requirements.txt` for complete list with versions.

## License

This project is designed for GSK equipment maintenance documentation processing. Please ensure compliance with your organization's policies when using with proprietary documents.

## Contributing

When contributing:
1. Follow SOLID principles
2. Add appropriate tests
3. Update documentation
4. Ensure backward compatibility
5. Add logging for debugging

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs for error details
3. Ensure configuration is correct
4. Verify all dependencies are installed 