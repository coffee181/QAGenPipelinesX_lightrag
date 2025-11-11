"""Example usage of the QA Generation Pipeline."""

import logging
from pathlib import Path

from dotenv import load_dotenv

from src.utils.config import ConfigManager
from src.services.progress_manager import ProgressManager
from src.services.pdf_processor import PDFProcessor
from src.services.question_service import QuestionService
from src.services.answer_service import AnswerService

# Optional implementation
try:
    from src.implementations.tesseract_ocr import TesseractOCR
    TESSERACT_AVAILABLE = True
except ImportError:
    TesseractOCR = None
    TESSERACT_AVAILABLE = False
from src.implementations.simple_text_chunker import SimpleTextChunker
from src.implementations.deepseek_question_generator import DeepSeekQuestionGenerator
from src.implementations.lightrag_rag import LightRAGImplementation
from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor


def setup_logging():
    """Setup basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def create_services(config_path: str = "config.yaml"):
    """Create and configure all services."""
    logger = setup_logging()
    
    # Load configuration
    config = ConfigManager(config_path)
    
    # Create progress manager
    progress_manager = ProgressManager(config)
    
    # Create implementations
    if TESSERACT_AVAILABLE:
        ocr = TesseractOCR(config)
        logger.info("TesseractOCR initialized successfully")
    else:
        ocr = None
        logger.warning("TesseractOCR not available - PDF processing will be disabled")
    
    text_chunker = SimpleTextChunker(config)
    
    question_generator = DeepSeekQuestionGenerator(config)
    
    rag = LightRAGImplementation(config)
    
    markdown_processor = SimpleMarkdownProcessor()
    
    # Create services
    pdf_processor = PDFProcessor(config, ocr, progress_manager)
    
    question_service = QuestionService(config, question_generator, text_chunker, markdown_processor, progress_manager)
    
    answer_service = AnswerService(rag, markdown_processor, progress_manager)
    
    return pdf_processor, question_service, answer_service, progress_manager, logger


def example_lightrag_answer_generation():
    """Example: Using existing LightRAG knowledge base for answer generation."""
    print("\n=== Example: LightRAG Answer Generation ===")
    
    logger = setup_logging()
    
    try:
        # Create custom configuration for LightRAG
        config = ConfigManager()
        
        # Set the LightRAG working directory to the existing knowledge base
        lightrag_working_dir = r"D:\Project\lightrag\selection_workdir"
        config.set('rag.lightrag.working_dir', lightrag_working_dir)
        
        logger.info(f"Using LightRAG working directory: {lightrag_working_dir}")
        
        # Create LightRAG implementation
        from src.implementations.lightrag_rag import LightRAGImplementation
        from src.implementations.simple_markdown_processor import SimpleMarkdownProcessor
        from src.services.progress_manager import ProgressManager
        from src.services.answer_service import AnswerService
        
        progress_manager = ProgressManager(config)
        rag = LightRAGImplementation(config)
        markdown_processor = SimpleMarkdownProcessor()
        answer_service = AnswerService(rag, markdown_processor, progress_manager)
        
        logger.info("✅ LightRAG services initialized successfully")
        
        # Get knowledge base statistics
        try:
            stats = rag.get_knowledge_base_stats()
            logger.info(f"📊 Knowledge base stats: {stats}")
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")
        
        # Test with some sample questions
        test_questions = [
            "这个文档主要讲了什么内容？",
            "有哪些重要的技术特性？",
            "文档中提到了哪些关键概念？"
        ]
        
        logger.info("🔍 Testing LightRAG query functionality:")
        
        for i, question in enumerate(test_questions, 1):
            try:
                logger.info(f"Question {i}: {question}")
                
                # Query using LightRAG
                response = rag.query_single_question(question)
                
                # Show first 150 characters of response
                response_preview = response[:150] + "..." if len(response) > 150 else response
                logger.info(f"  ✅ Response: {response_preview}")
                
                if i >= 2:  # Limit to 2 questions to save time
                    break
                    
            except Exception as e:
                logger.error(f"Question {i} failed: {e}")
        
        # Test batch question processing if we have question files
        questions_dir = Path("batch_output/questions")
        if questions_dir.exists():
            logger.info("🔄 Testing batch answer generation:")
            
            question_files = list(questions_dir.glob("*questions.jsonl"))
            if question_files:
                output_dir = Path("lightrag_output/qa")
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Process first question file as example
                first_file = question_files[0]
                logger.info(f"Processing questions from: {first_file.name}")
                
                try:
                    qa_result = answer_service.generate_answers_for_questions(
                        first_file,
                        output_dir / f"lightrag_{first_file.stem.replace('_questions', '')}_qa.jsonl"
                    )
                    logger.info(f"✅ Generated {len(qa_result.qa_pairs)} QA pairs using LightRAG")
                    
                except Exception as e:
                    logger.error(f"Batch processing failed: {e}")
            else:
                logger.info("No question files found for batch processing")
        
        logger.info("🎉 LightRAG answer generation test completed!")
        
    except Exception as e:
        logger.error(f"LightRAG answer generation test failed: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")


def example_single_pdf_processing():
    """Example: Process a single PDF through the complete pipeline."""
    print("=== Example: Single PDF Processing ===")
    
    # Create services
    pdf_processor, question_service, answer_service, progress_manager, logger = create_services()
    
    # Define paths
    pdf_path = Path("example_manual.pdf")  # Replace with your PDF
    output_dir = Path("example_output")
    
    # Create output directories
    texts_dir = output_dir / "texts"
    questions_dir = output_dir / "questions"
    qa_dir = output_dir / "qa"
    
    for dir_path in [texts_dir, questions_dir, qa_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        # Step 1: Process PDF to text
        logger.info("Step 1: Processing PDF...")
        if pdf_path.exists():
            pdf_result = pdf_processor.process_pdf(pdf_path, texts_dir)
            logger.info(f"PDF processed: {pdf_result.document_id}")
            
            # Step 2: Generate questions
            logger.info("Step 2: Generating questions...")
            text_file = texts_dir / f"{pdf_result.document_id}.txt"
            question_result = question_service.generate_questions_for_document(
                text_file, questions_dir
            )
            logger.info(f"Generated {len(question_result.questions)} questions")
            
            # Step 3: Setup knowledge base and generate answers
            logger.info("Step 3: Setting up knowledge base...")
            answer_service.setup_knowledge_base(texts_dir)
            
            logger.info("Step 4: Generating answers...")
            questions_file = questions_dir / f"{pdf_result.document_id}_questions.jsonl"
            qa_result = answer_service.generate_answers_for_questions(
                questions_file, qa_dir / f"{pdf_result.document_id}_qa.jsonl"
            )
            logger.info(f"Generated {len(qa_result.qa_pairs)} QA pairs")
            
            logger.info("Single PDF processing completed successfully!")
            
        else:
            logger.warning(f"PDF file not found: {pdf_path}")
            logger.info("Please place a PDF file named 'example_manual.pdf' in the current directory")
            
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")


def example_batch_processing():
    """Example: Batch process multiple PDFs."""
    print("\n=== Example: Batch Processing ===")
    
    # Create services
    pdf_processor, question_service, answer_service, progress_manager, logger = create_services()
    
    # Define paths
    pdf_dir = Path("example_pdfs")  # Directory with multiple PDFs
    output_dir = Path("batch_output")
    
    # Create output directories
    texts_dir = output_dir / "texts"
    questions_dir = output_dir / "questions"
    qa_dir = output_dir / "qa"
    
    for dir_path in [texts_dir, questions_dir, qa_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    try:
        if pdf_dir.exists() and any(pdf_dir.glob("*.pdf")):
            # Step 1: Process all PDFs
            logger.info("Step 1: Processing PDFs...")
            pdf_results = pdf_processor.process_directory(pdf_dir, texts_dir)
            logger.info(f"Processed {len(pdf_results)} PDFs")
            
            # Step 2: Generate questions for all documents
            logger.info("Step 2: Generating questions...")
            question_results = question_service.generate_questions_for_directory(
                texts_dir, questions_dir
            )
            logger.info(f"Generated questions for {len(question_results)} documents")
            
            # Step 3: Setup knowledge base and generate answers
            logger.info("Step 3: Setting up knowledge base...")
            answer_service.setup_knowledge_base(texts_dir)
            
            logger.info("Step 4: Generating answers...")
            answer_results = answer_service.generate_answers_for_directory(
                questions_dir, qa_dir
            )
            logger.info(f"Generated answers for {len(answer_results)} question files")
            
            logger.info("Batch processing completed successfully!")
            
        else:
            logger.warning(f"PDF directory not found or empty: {pdf_dir}")
            logger.info("Please create a directory named 'example_pdfs' with PDF files")
            
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")


def example_progress_monitoring():
    """Example: Monitor progress of operations."""
    print("\n=== Example: Progress Monitoring ===")
    
    # Create services
    _, _, _, progress_manager, logger = create_services()
    
    try:
        # List all sessions
        sessions = progress_manager.list_sessions()
        logger.info(f"Found {len(sessions)} sessions:")
        
        for session in sessions:
            stats = progress_manager.get_session_stats(session["session_id"])
            logger.info(f"  Session: {session['session_id']}")
            logger.info(f"    Status: {session['status']}")
            logger.info(f"    Progress: {stats.get('progress_percentage', 0):.1f}%")
            logger.info(f"    Operation: {session['operation_type']}")
            logger.info(f"    Created: {session['start_time']}")
            
            if session["status"] == "failed":
                logger.info(f"    Error: {session.get('error_message', 'Unknown error')}")
            
            logger.info("")
            
    except Exception as e:
        logger.error(f"Progress monitoring failed: {str(e)}")


def example_custom_configuration():
    """Example: Using custom configuration."""
    print("\n=== Example: Custom Configuration ===")
    
    logger = setup_logging()
    
    try:
        # Create custom configuration
        config = ConfigManager()
        
        # Override specific settings
        config.set('text_chunker.max_chunk_size', 500)  # Smaller chunks
        config.set('text_chunker.overlap_size', 100)
        
        # Show current configuration
        logger.info("Current configuration:")
        logger.info(f"  Chunk size: {config.get('text_chunker.max_chunk_size')}")
        logger.info(f"  Overlap size: {config.get('text_chunker.overlap_size')}")
        logger.info(f"  DeepSeek model: {config.get('question_generator.deepseek.model')}")
        logger.info(f"  RAG working dir: {config.get('rag.lightrag.working_dir')}")
        
    except Exception as e:
        logger.error(f"Configuration example failed: {str(e)}")


def main():
    """Run all examples."""
    load_dotenv()
    print("QA Generation Pipeline - Example Usage")
    print("=" * 50)
    
    # Run examples
    example_lightrag_answer_generation()
    example_single_pdf_processing()
    example_batch_processing()
    example_progress_monitoring()
    example_custom_configuration()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo run the examples with real data:")
    print("1. Place PDF files in the current directory or 'example_pdfs' folder")
    print("2. Ensure your DeepSeek API key is configured")
    print("3. Run: python example_usage.py")


if __name__ == "__main__":
    main() 