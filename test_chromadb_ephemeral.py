"""Test ChromaDB with EphemeralClient to bypass platform.system() issue"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
os.environ['CHROMA_DB_IMPL'] = 'sqlite'

import chromadb

print("Testing EphemeralClient...")
try:
    client = chromadb.EphemeralClient()
    print("[OK] EphemeralClient created successfully")
    
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
    print("\n[SUCCESS] EphemeralClient works! This avoids platform.system() issue.")
    
except Exception as e:
    print(f"[FAILED] EphemeralClient error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("Testing PersistentClient...")
try:
    client = chromadb.PersistentClient(path="./test_db")
    print("[FAILED] PersistentClient created - should have crashed on platform.system()")
except Exception as e:
    print(f"[EXPECTED] PersistentClient failed with: {type(e).__name__}")
    print(f"Error: {str(e)[:100]}")
