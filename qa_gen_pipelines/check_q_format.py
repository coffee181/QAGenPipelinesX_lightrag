#!/usr/bin/env python3
"""Check the format of the question file."""

import json
from pathlib import Path

def check_question_file(file_path):
    """Check the format of a JSONL question file."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"File does not exist: {file_path}")
        return
    
    print(f"Checking file: {file_path}")
    print(f"File size: {file_path.stat().st_size} bytes")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        print(f"Total lines: {len(lines)}")
        
        # Check first 3 lines
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if not line:
                print(f"Line {i+1}: Empty line")
                continue
                
            try:
                data = json.loads(line)
                print(f"Line {i+1}: Valid JSON - Keys: {list(data.keys())}")
                
                # Check if it has the expected format
                if "question_id" in data and ("text" in data or "content" in data):
                    print(f"  -> Standard format detected")
                elif "messages" in data:
                    print(f"  -> Messages format detected")
                else:
                    print(f"  -> Unknown format")
                    
            except json.JSONDecodeError as e:
                print(f"Line {i+1}: Invalid JSON - {e}")
                print(f"  Content preview: {line[:100]}...")
                
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    check_question_file("D:/Project/lightrag/q.jsonl") 