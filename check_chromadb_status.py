"""Check if ChromaDB collection exists and has data"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Mock platform functions
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

import chromadb
from pathlib import Path

db_path = Path(__file__).parent / "rag_system" / "chroma_db"

try:
    print(f"Checking ChromaDB at: {db_path}")
    client = chromadb.PersistentClient(path=str(db_path))
    
    # List collections
    collections = client.list_collections()
    print(f"\nFound {len(collections)} collection(s):")
    
    for coll in collections:
        print(f"\n  Collection: {coll.name}")
        count = coll.count()
        print(f"  Document count: {count}")
        
        if count > 0:
            # Sample a few items
            results = coll.peek(limit=3)
            print(f"  Sample IDs: {results['ids'][:3]}")
            if results['metadatas']:
                print(f"  Sample metadata: {results['metadatas'][0]}")
        else:
            print("  [EMPTY - No documents indexed yet]")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
