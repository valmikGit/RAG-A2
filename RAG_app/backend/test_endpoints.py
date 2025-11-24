"""
Test script for RAG Backend API endpoints
Usage: python test_endpoints.py
"""
import requests
import json
from typing import Dict, Any

# Backend URL
BASE_URL = "http://localhost:8000"

def print_response(endpoint: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*70}")
    print(f"Endpoint: {endpoint}")
    print(f"Status Code: {response.status_code}")
    print(f"{'='*70}")
    
    try:
        data = response.json()
        print(json.dumps(data, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"Response Text: {response.text}")
        print(f"Error parsing JSON: {e}")
    
    print(f"{'='*70}\n")

def test_health():
    """Test /health endpoint"""
    print("\n🏥 Testing /health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("GET /health", response)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Health check passed!")
            
            # Verify key components
            details = data.get("details", {})
            if details.get("chroma_present"):
                print(f"   ✓ ChromaDB: Connected")
                print(f"   ✓ Collection: {details.get('collection_name')}")
                print(f"   ✓ Document count: {details.get('collection_count')}")
                print(f"   ✓ Database path: {details.get('chroma_path')}")
            else:
                print("   ✗ ChromaDB: NOT connected")
                
            if details.get("gemini_initialized"):
                print(f"   ✓ Gemini: Initialized")
                print(f"   ✓ Model: {details.get('gemini_model')}")
            else:
                print("   ✗ Gemini: NOT initialized")
                
            # Check sample chunk for artifacts
            if "sample_chunk_preview" in details:
                sample = details["sample_chunk_preview"]
                print(f"\n   📄 Sample chunk: {sample[:100]}...")
                
                # Check for artifacts
                import re
                if re.search(r'\d+\s+Julius\s+Caesar\s+ACT', sample):
                    print("   ⚠️  WARNING: Sample chunk contains artifacts!")
                elif re.search(r'ACT\s+\d+\.\s+SC\.\s+\d+', sample):
                    print("   ⚠️  WARNING: Sample chunk contains ACT/SCENE markers!")
                else:
                    print("   ✓ Sample chunk appears clean")
                    
                if "sample_metadata" in details:
                    meta = details["sample_metadata"]
                    if meta.get("act"):
                        print(f"   ✓ Metadata has Act/Scene: Act {meta.get('act')}, Scene {meta.get('scene')}")
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing /health: {e}")

def test_collections():
    """Test /collections endpoint"""
    print("\n📚 Testing /collections endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/collections")
        print_response("GET /collections", response)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Collections endpoint accessible!")
            
            collections = data.get("collections", [])
            print(f"\n   Found {len(collections)} collection(s):")
            for col in collections:
                active = "🟢 ACTIVE" if col.get("is_active") else ""
                print(f"   - {col['name']} (count: {col.get('count', 'N/A')}) {active}")
                
            print(f"\n   Active collection: {data.get('active_collection')}")
            print(f"   Database path: {data.get('database_path')}")
        else:
            print(f"❌ Collections check failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error testing /collections: {e}")

def test_query():
    """Test /query endpoint with sample question"""
    print("\n🔍 Testing /query endpoint...")
    
    test_questions = [
        "What is the relationship between Brutus and Caesar?",
        "Who is involved in the conspiracy?",
        "What are the main themes in Julius Caesar?"
    ]
    
    for question in test_questions:
        print(f"\n   Question: {question}")
        try:
            response = requests.post(
                f"{BASE_URL}/query",
                json={"user_query": question}
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Answer received ({len(data['answer'])} chars)")
                print(f"   📄 Answer: {data['answer'][:200]}...")
                print(f"   📚 Contexts: {len(data['contexts'])} chunks retrieved")
                
                # Check first context for artifacts
                if data['contexts']:
                    first_context = data['contexts'][0]
                    context_text = first_context.get('text', first_context) if isinstance(first_context, dict) else first_context
                    context_meta = first_context.get('metadata', {}) if isinstance(first_context, dict) else {}
                    
                    print(f"\n   First context (first 150 chars):")
                    print(f"   Text: {context_text[:150]}...")
                    
                    if context_meta:
                        print(f"   Metadata:")
                        if context_meta.get('act'):
                            print(f"     - Act: {context_meta.get('act')}, Scene: {context_meta.get('scene')}")
                        print(f"     - Page: {context_meta.get('page_number', 'N/A')}")
                        print(f"     - Cleaned: {context_meta.get('cleaned', False)}")
                    
                    import re
                    if re.search(r'\d+\s+Julius\s+Caesar\s+ACT', context_text):
                        print("   ⚠️  WARNING: Context contains page number + ACT artifacts!")
                    elif re.search(r'ACT\s+\d+\.\s+SC\.\s+\d+', context_text):
                        print("   ⚠️  WARNING: Context contains ACT/SCENE markers!")
                    else:
                        print("   ✓ Context appears clean")
            else:
                print(f"   ❌ Query failed with status {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
        
        print()

def main():
    """Run all tests"""
    print("="*70)
    print("RAG BACKEND API TESTING")
    print("="*70)
    print(f"Testing backend at: {BASE_URL}")
    print("Make sure the backend is running: python app.py or docker-compose up")
    print()
    
    # Check if backend is reachable
    try:
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        print("✅ Backend is reachable!")
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to backend!")
        print("   Make sure the backend is running on port 8000")
        print("   Run: cd RAG_app/backend && python app.py")
        return
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return
    
    # Run tests
    test_health()
    test_collections()
    test_query()
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
