#!/usr/bin/env python3
"""
Cleanup script to remove old FAISS-related files.
Run this to clean up after migrating to Qdrant.
"""

import os
from pathlib import Path

# Files to remove
OLD_FILES = [
    "build_index.py",          # Original FAISS index builder
    "serve_retriever.py",      # Original FAISS retriever
    "search_once.py",          # Original FAISS search
]

# Index files to remove
OLD_INDEX_FILES = [
    "rag_prep/index/faiss.index",
    "rag_prep/index/index_info.json", 
    "rag_prep/index/meta.jsonl",
]

def main():
    print("🧹 Cleaning up old FAISS-related files...")
    
    # Remove Python files
    removed_files = 0
    for file_path in OLD_FILES:
        if Path(file_path).exists():
            os.remove(file_path)
            print(f"✅ Removed: {file_path}")
            removed_files += 1
        else:
            print(f"⚠️  Already gone: {file_path}")
    
    # Remove old index files (optional - ask user)
    print(f"\nFound {len(OLD_INDEX_FILES)} old index files:")
    for index_file in OLD_INDEX_FILES:
        if Path(index_file).exists():
            print(f"  - {index_file}")
    
    if any(Path(f).exists() for f in OLD_INDEX_FILES):
        response = input("\nRemove old FAISS index files? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            for index_file in OLD_INDEX_FILES:
                if Path(index_file).exists():
                    os.remove(index_file)
                    print(f"✅ Removed: {index_file}")
                    removed_files += 1
        else:
            print("ℹ️  Keeping old index files (you can remove them manually)")
    
    print(f"\n🎉 Cleanup complete! Removed {removed_files} files.")
    
    if removed_files > 0:
        print("\n📝 Next steps:")
        print("1. Make sure your QDRANT_URL and QDRANT_API_KEY are set")
        print("2. Run: make rebuild")
        print("3. Run: make retriever")
        print("4. Test: make answer QUESTION=\"your question\"")

if __name__ == "__main__":
    main()