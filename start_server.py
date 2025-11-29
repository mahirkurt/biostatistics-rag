"""
Simple API Server Starter with Model Preloading
"""
import os
import sys
import time

# Fix encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Add path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print("\n" + "="*60)
    print("BIOSTATISTICS RAG API SERVER")
    print("="*60)
    
    # Preload models BEFORE starting server
    print("\n[1/3] Preloading RAG system...")
    start = time.time()
    
    from rag_system.query_interface import BiostatisticsRAG
    rag = BiostatisticsRAG(verbose=True)
    
    print(f"\n[2/3] Models loaded in {time.time()-start:.1f}s")
    
    # Inject preloaded RAG into api_server
    import api_server
    api_server.rag_system = rag
    
    print("[3/3] Starting HTTP server...")
    print("\n" + "="*60)
    print("Server ready at http://localhost:8002")
    print("="*60 + "\n")
    
    import uvicorn
    
    try:
        uvicorn.run(
            api_server.app,
            host="127.0.0.1",
            port=8002,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        print("\nServer stopped")
    except Exception as e:
        print(f"\nError: {e}")
        input("Press Enter to exit...")
