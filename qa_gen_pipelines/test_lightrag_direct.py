"""Direct test of LightRAG query functionality."""

import os
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv


async def test_lightrag_direct_query():
    """Test LightRAG query directly on existing knowledge base."""
    
    # Load environment variables
    load_dotenv()
    
    print("🔧 Testing LightRAG direct query on existing knowledge base")
    
    try:
        # Import LightRAG
        from lightrag import LightRAG, QueryParam
        
        print("✅ LightRAG imported successfully")
        
        # Set the working directory to the existing knowledge base
        working_dir = r"D:\Project\lightrag\selection_workdir"
        print(f"📁 Using working directory: {working_dir}")
        
        # Check if directory exists and has data
        wd_path = Path(working_dir)
        if not wd_path.exists():
            print(f"❌ Working directory does not exist: {working_dir}")
            return
        
        # List files in working directory
        files = list(wd_path.glob("*"))
        print(f"📊 Found {len(files)} files in working directory:")
        for file in files[:5]:  # Show first 5 files
            size_mb = file.stat().st_size / (1024 * 1024) if file.is_file() else 0
            print(f"  - {file.name}: {size_mb:.2f} MB")
        
        # Check API keys
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        
        print(f"🔑 DeepSeek API key: {'✅ Set' if deepseek_key else '❌ Not set'}")
        print(f"🔑 OpenAI API key: {'✅ Set' if openai_key else '❌ Not set'}")
        
        if not deepseek_key:
            print("⚠️ Warning: No DeepSeek API key found.")
            print("Without API keys, LightRAG cannot initialize properly.")
            print("However, let's try to access the existing data files directly...")
            
            # Try to read some data from the existing files
            try:
                import json
                
                # Try to read the full docs cache
                full_docs_path = wd_path / "kv_store_full_docs.json"
                if full_docs_path.exists():
                    with open(full_docs_path, 'r', encoding='utf-8') as f:
                        full_docs = json.load(f)
                    print(f"📚 Found {len(full_docs)} documents in full docs cache")
                    
                    # Show some document info
                    for i, (doc_id, doc_content) in enumerate(list(full_docs.items())[:2]):
                        content_preview = doc_content[:100] + "..." if len(doc_content) > 100 else doc_content
                        print(f"  Document {i+1}: {content_preview}")
                
                # Try to read text chunks
                chunks_path = wd_path / "kv_store_text_chunks.json" 
                if chunks_path.exists():
                    with open(chunks_path, 'r', encoding='utf-8') as f:
                        chunks = json.load(f)
                    print(f"📝 Found {len(chunks)} text chunks")
                
                # Try to read LLM response cache (this shows what queries have been made)
                cache_path = wd_path / "kv_store_llm_response_cache.json"
                if cache_path.exists():
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cache = json.load(f)
                    print(f"💾 Found {len(cache)} cached LLM responses")
                    
                    # Show some example queries
                    print("📋 Some example cached queries:")
                    for i, (cache_key, response) in enumerate(list(cache.items())[:3]):
                        # The cache key might contain the query
                        if len(cache_key) < 200:  # Only show short keys
                            print(f"  Query {i+1}: {cache_key[:100]}...")
                
            except Exception as e:
                print(f"❌ Failed to read data files: {e}")
            
            return
        
        # Define simple LLM and embedding functions for basic functionality
        print("\n🚀 Creating LightRAG with minimal configuration...")
        
        # Simple LLM function using OpenAI API format
        async def simple_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs):
            """Simple LLM function for testing."""
            import openai
            
            # Configure OpenAI client for DeepSeek
            client = openai.AsyncOpenAI(
                api_key=deepseek_key,
                base_url="https://api.deepseek.com/v1"
            )
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            for msg in history_messages:
                messages.append(msg)
            
            messages.append({"role": "user", "content": prompt})
            
            response = await client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                **kwargs
            )
            
            return response.choices[0].message.content
        
        # Simple embedding function
        async def simple_embedding_func(texts):
            """Simple embedding function for testing."""
            import openai
            import numpy as np
            
            if openai_key:
                # Use OpenAI embeddings
                client = openai.AsyncOpenAI(api_key=openai_key)
                
                response = await client.embeddings.create(
                    model="text-embedding-3-large",  # Use large model for 3072 dimensions
                    input=texts
                )
                
                embeddings = [data.embedding for data in response.data]
                return np.array(embeddings)
            else:
                # Simple fallback - create 3072 dimensional embeddings
                import hashlib
                embeddings = []
                for text in texts:
                    hash_obj = hashlib.md5(text.encode())
                    hash_int = int(hash_obj.hexdigest(), 16)
                    embedding = [(hash_int >> i) & 1 for i in range(3072)]  # 3072 dimensions
                    embeddings.append(embedding)
                return np.array(embeddings, dtype=np.float32)
        
        # Try to initialize LightRAG with functions
        try:
            from lightrag.utils import EmbeddingFunc
            
            rag = LightRAG(
                working_dir=working_dir,
                llm_model_func=simple_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=3072,  # Match the existing knowledge base
                    max_token_size=8192,
                    func=simple_embedding_func
                )
            )
            
            print("✅ LightRAG initialized with custom functions")
            
            # Test a simple query
            test_question = "这个文档主要讲了什么内容？"
            print(f"\n🔍 Testing query: {test_question}")
            
                        try:                # Try different query modes                modes_to_try = ["local", "global", "hybrid"]                                for mode in modes_to_try:                    try:                        print(f"  Trying {mode} mode...")                        response = await rag.aquery(test_question, param=QueryParam(mode=mode))                        print(f"✅ Query successful using {mode} mode!")                        print(f"📄 Response ({len(response)} chars): {response[:300]}...")                        break                    except Exception as mode_error:                        print(f"  ❌ {mode} mode failed: {mode_error}")                        continue                else:                    print("❌ All query modes failed")                            except Exception as e:                print(f"❌ Query failed: {e}")                import traceback                print(f"Detailed error: {traceback.format_exc()}")
                
        except Exception as e:
            print(f"❌ Failed to initialize LightRAG with functions: {e}")
            
    except ImportError as e:
        print(f"❌ Failed to import LightRAG: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")


def main():
    """Run the test."""
    asyncio.run(test_lightrag_direct_query())


if __name__ == "__main__":
    main() 