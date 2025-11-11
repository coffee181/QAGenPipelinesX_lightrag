"""Test LightRAG with proper API configuration."""

import os
from pathlib import Path
from dotenv import load_dotenv


def test_lightrag_with_existing_data():
    """Test LightRAG using existing knowledge base with proper API configuration."""
    
    # Load environment variables
    load_dotenv()
    
    print("🔧 Testing LightRAG with existing knowledge base")
    
    try:
        # Import our implementation
        from src.utils.config import ConfigManager
        from src.implementations.lightrag_rag import LightRAGImplementation
        
        print("✅ Imports successful")
        
        # Create configuration
        config = ConfigManager()
        
        # Set the working directory to the existing knowledge base
        lightrag_working_dir = r"D:\Project\lightrag\selection_workdir"
        config.set('rag.lightrag.working_dir', lightrag_working_dir)
        
        print(f"📁 Using LightRAG working directory: {lightrag_working_dir}")
        
        # Check if directory exists and has data
        wd_path = Path(lightrag_working_dir)
        if not wd_path.exists():
            print(f"❌ Working directory does not exist: {lightrag_working_dir}")
            return
        
        # List files in working directory
        files = list(wd_path.glob("*"))
        print(f"📊 Found {len(files)} files in working directory:")
        for file in files[:5]:  # Show first 5 files
            size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
            print(f"  - {file.name}: {size_mb:.2f} MB")
        
        # Check API keys
        deepseek_key = config.get("question_generator.deepseek.api_key") or os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        print(f"🔑 DeepSeek API key: {'✅ Set' if deepseek_key else '❌ Not set'}")
        print(f"🔑 OpenAI API key: {'✅ Set' if openai_key else '❌ Not set'}")
        
        if not deepseek_key:
            print("⚠️ Warning: No DeepSeek API key found. LightRAG may not work properly.")
            print("Please set DEEPSEEK_API_KEY environment variable or configure it in config.yaml")
        
        # Initialize LightRAG
        print("\n🚀 Initializing LightRAG...")
        
        try:
            rag = LightRAGImplementation(config)
            print("✅ LightRAG initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize LightRAG: {e}")
            return
        
        # Get knowledge base statistics
        try:
            stats = rag.get_knowledge_base_stats()
            print(f"📊 Knowledge base stats: {stats}")
        except Exception as e:
            print(f"⚠️ Could not get stats: {e}")
        
        # Test queries only if we have API keys
        if deepseek_key:
            print("\n🔍 Testing LightRAG query functionality:")
            
            test_questions = [
                "这个文档主要讲了什么内容？",
                "有哪些重要的技术特性？"
            ]
            
            for i, question in enumerate(test_questions, 1):
                try:
                    print(f"\nQuestion {i}: {question}")
                    
                    # Query using our implementation
                    response = rag.query_single_question(question)
                    
                    # Show response preview
                    response_preview = response[:200] + "..." if len(response) > 200 else response
                    print(f"✅ Response ({len(response)} chars): {response_preview}")
                    
                    # Only test one question for now
                    if i >= 1:
                        break
                        
                except Exception as e:
                    print(f"❌ Question {i} failed: {e}")
        else:
            print("\n⚠️ Skipping query tests due to missing API keys")
        
        print("\n🎉 LightRAG test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    test_lightrag_with_existing_data() 