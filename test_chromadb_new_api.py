"""Test ChromaDB 1.3.5 with new API"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

import chromadb

print("ChromaDB version:", chromadb.__version__)
print("\nTesting Client() with ephemeral mode...")
try:
    # New API: Client() is ephemeral by default
    client = chromadb.Client()
    print("[OK] Client() created successfully (ephemeral)")
    
    collection = client.create_collection(name="test", metadata={"hnsw:space": "cosine"})
    print("[OK] Collection created")
    
    collection.add(
        documents=["test document"],
        ids=["test1"],
        metadatas=[{"source": "test"}],
        embeddings=[[0.1] * 1024]
    )
    print("[OK] Document added")
    
    results = collection.query(
        query_embeddings=[[0.1] * 1024],
        n_results=1
    )
    print(f"[OK] Query results: {results['ids']}")
    print("\n[SUCCESS] New Client() API works!")
    
except Exception as e:
    print(f"[FAILED] Client() error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing PersistentClient() with path...")
try:
    client = chromadb.PersistentClient(path="./test_db")
    print("[OK or FAILED?] PersistentClient created")
except Exception as e:
    print(f"[EXPECTED FAIL] PersistentClient: {type(e).__name__}: {str(e)[:150]}")
