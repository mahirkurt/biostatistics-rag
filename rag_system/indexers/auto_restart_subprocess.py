"""
Auto-restart indexer using subprocess
Automatically restarts after interrupts until all PDFs are indexed
"""

import subprocess
import sys
import time
import os

# Set environment before any imports
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Mock platform
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

import chromadb
from pathlib import Path

CHROMA_DB_PATH = Path(__file__).parent.parent / "chroma_db"
TEXTBOOKS_DIR = Path(__file__).parent.parent.parent / "textbooks"
COLLECTION_NAME = "biostatistics_main"
INDEXER_SCRIPT = Path(__file__).parent / "run_indexer_pure_onnx.py"


def get_doc_count():
    """Get current document count from ChromaDB."""
    try:
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(COLLECTION_NAME)
        return collection.count()
    except Exception:
        return 0


def get_pdf_count():
    """Get total PDF count."""
    return len(list(TEXTBOOKS_DIR.glob("*.pdf")))


def main():
    print("=" * 50)
    print("Auto-Restart Indexer (Subprocess Mode)")
    print("=" * 50)
    
    total_pdfs = get_pdf_count()
    print(f"Total PDFs: {total_pdfs}")
    
    max_attempts = 200
    attempt = 0
    last_count = get_doc_count()
    no_progress_count = 0
    
    print(f"Starting doc count: {last_count}")
    print()
    
    while attempt < max_attempts:
        attempt += 1
        print(f"[Attempt {attempt}] Running indexer...")
        
        try:
            # Run the indexer as a subprocess
            result = subprocess.run(
                [sys.executable, str(INDEXER_SCRIPT)],
                capture_output=False,  # Let output go to console
                timeout=600  # 10 minute timeout per attempt
            )
            
            if result.returncode == 0:
                print("\n[SUCCESS] Indexer completed normally!")
                break
            
        except subprocess.TimeoutExpired:
            print("\n[TIMEOUT] Attempt timed out, restarting...")
        except Exception as e:
            print(f"\n[ERROR] {e}")
        
        # Check progress
        current_count = get_doc_count()
        progress = current_count - last_count
        
        print(f"\n[PROGRESS] Docs: {last_count} → {current_count} (+{progress})")
        
        if progress == 0:
            no_progress_count += 1
            if no_progress_count >= 5:
                print("[WARNING] No progress for 5 attempts. There may be an issue.")
        else:
            no_progress_count = 0
        
        last_count = current_count
        
        # Brief pause before restart
        print("[RESTART] Restarting in 3 seconds...\n")
        time.sleep(3)
    
    final_count = get_doc_count()
    print()
    print("=" * 50)
    print(f"Final document count: {final_count}")
    print(f"Total attempts: {attempt}")
    print("=" * 50)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[STOPPED] Manual interrupt. Progress saved.")
        print(f"Current doc count: {get_doc_count()}")
