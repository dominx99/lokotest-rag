#!/usr/bin/env python3
"""
Test script for Qdrant integration.
Validates that the migration and new components work correctly.
"""

import os
import sys
import time
import subprocess
import requests
from pathlib import Path

def check_qdrant_connection():
    """Check if Qdrant is running and accessible."""
    print("🔍 Checking Qdrant connection...")
    
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = os.environ.get("QDRANT_PORT", "6333")
    url = f"http://{host}:{port}/health"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            print(f"✅ Qdrant is running at {host}:{port}")
            return True
    except Exception as e:
        print(f"❌ Qdrant connection failed: {e}")
        print("💡 Make sure Qdrant is running:")
        print("   docker run -p 6333:6333 qdrant/qdrant")
        return False

def check_collection_exists():
    """Check if the Qdrant collection exists."""
    print("\n🔍 Checking Qdrant collection...")
    
    try:
        from qdrant_store import get_qdrant_store
        
        collection_name = os.environ.get("QDRANT_COLLECTION", "rag_documents")
        qdrant_store = get_qdrant_store(collection_name)
        
        info = qdrant_store.get_collection_info()
        if info:
            points_count = info.get("points_count", 0)
            print(f"✅ Collection '{collection_name}' exists with {points_count} points")
            return True, points_count
        else:
            print(f"❌ Collection '{collection_name}' not found")
            return False, 0
            
    except Exception as e:
        print(f"❌ Collection check failed: {e}")
        return False, 0

def test_direct_search():
    """Test direct Qdrant search functionality."""
    print("\n🔍 Testing direct search...")
    
    try:
        import numpy as np
        from qdrant_store import get_qdrant_store
        
        collection_name = os.environ.get("QDRANT_COLLECTION", "rag_documents")
        qdrant_store = get_qdrant_store(collection_name)
        
        # Get collection info to determine vector size
        info = qdrant_store.get_collection_info()
        vector_size = info.get("config", {}).get("vector_size", 3072)
        
        # Create a random test vector
        test_vector = np.random.rand(vector_size).astype(np.float32)
        test_vector = test_vector / np.linalg.norm(test_vector)
        
        results = qdrant_store.search(test_vector, limit=3)
        
        if results:
            print(f"✅ Direct search works - found {len(results)} results")
            print(f"   Sample result ID: {results[0]['chunk_id']}")
            return True
        else:
            print("❌ Direct search returned no results")
            return False
            
    except Exception as e:
        print(f"❌ Direct search failed: {e}")
        return False

def test_retriever_service():
    """Test the Qdrant retriever service."""
    print("\n🔍 Testing retriever service...")
    
    # Start the service in background
    service_cmd = [sys.executable, "serve_qdrant_retriever.py"]
    
    try:
        print("   Starting retriever service...")
        process = subprocess.Popen(
            service_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait a moment for service to start
        time.sleep(3)
        
        # Test the service
        test_url = "http://127.0.0.1:8000/search?q=test"
        response = requests.get(test_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            hits = data.get("hits", [])
            print(f"✅ Retriever service works - found {len(hits)} hits")
            
            if hits:
                print(f"   Sample hit: {hits[0].get('chunk_id')}")
            
            success = True
        else:
            print(f"❌ Retriever service returned status {response.status_code}")
            success = False
            
    except Exception as e:
        print(f"❌ Retriever service test failed: {e}")
        success = False
    finally:
        # Clean up the process
        try:
            process.terminate()
            process.wait(timeout=5)
        except:
            process.kill()
    
    return success

def check_required_files():
    """Check if all required files are present."""
    print("\n🔍 Checking required files...")
    
    required_files = [
        "qdrant_store.py",
        "build_qdrant_index.py", 
        "serve_qdrant_retriever.py",
        "search_once_qdrant.py",
        "migrate_to_qdrant.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    else:
        print("✅ All required files present")
        return True

def run_integration_test():
    """Run complete integration test."""
    print("🚀 Qdrant Integration Test")
    print("=" * 40)
    
    all_passed = True
    
    # Check files
    all_passed &= check_required_files()
    
    # Check Qdrant connection
    all_passed &= check_qdrant_connection()
    
    # Check collection
    collection_exists, points_count = check_collection_exists()
    all_passed &= collection_exists
    
    if not collection_exists:
        print("\n💡 To create/migrate collection:")
        print("   python migrate_to_qdrant.py  # or")
        print("   python build_qdrant_index.py")
        return False
    
    # Test search functionality
    if points_count > 0:
        all_passed &= test_direct_search()
        all_passed &= test_retriever_service()
    else:
        print("⚠️  Collection is empty - skipping search tests")
    
    print("\n" + "=" * 40)
    if all_passed:
        print("🎉 All tests passed! Qdrant integration is working correctly.")
        print("\n📖 Next steps:")
        print("   1. Start retriever: python serve_qdrant_retriever.py")
        print("   2. Test search: python search_once_qdrant.py")
        print("   3. Use RAG: python answer_rag.py 'your question'")
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False
    
    return True

if __name__ == "__main__":
    success = run_integration_test()
    sys.exit(0 if success else 1)