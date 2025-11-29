"""
Restart-Based Indexer
=====================
Processes PDFs in batches of 5, restarting between batches
to avoid long-running process issues on this system

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os
import sys
from pathlib import Path
import subprocess
import json
import time

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
TEXTBOOKS_DIR = BASE_DIR / "textbooks"
STATE_FILE = BASE_DIR / "indexing_state.json"
WORKER_SCRIPT = Path(__file__).parent / "indexer_worker.py"

def load_state():
    """Load indexing state"""
    if STATE_FILE.exists():
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    return {"processed": [], "failed": [], "total_chunks": 0}

def save_state(state):
    """Save indexing state"""
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2)

def main():
    print("=" * 80)
    print("BIOSTATISTICS RAG - RESTART-BASED INDEXER")
    print("=" * 80)
    print("Processing in batches of 5 PDFs with process restart")
    print()
    
    # Get all PDFs
    pdf_files = sorted(TEXTBOOKS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("ERROR: No PDFs found")
        return
    
    # Load state
    state = load_state()
    processed_set = set(state["processed"])
    
    # Find remaining PDFs
    remaining = [pdf for pdf in pdf_files if pdf.name not in processed_set]
    
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Already processed: {len(processed_set)}")
    print(f"Remaining: {len(remaining)}")
    print(f"Total chunks so far: {state['total_chunks']}")
    print()
    
    if not remaining:
        print("All PDFs already processed!")
        print(f"\nFinal stats:")
        print(f"  Total chunks: {state['total_chunks']}")
        print(f"  Successful: {len(state['processed'])}")
        print(f"  Failed: {len(state['failed'])}")
        return
    
    # Process in batches of 5
    BATCH_SIZE = 5
    
    for batch_idx in range(0, len(remaining), BATCH_SIZE):
        batch = remaining[batch_idx:batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"{'='*80}")
        print(f"BATCH {batch_num}/{total_batches} - Processing {len(batch)} PDFs")
        print(f"{'='*80}")
        
        for pdf in batch:
            print(f"  - {pdf.name}")
        print()
        
        # Create batch file list
        batch_file = BASE_DIR / f"batch_{batch_num}.txt"
        with open(batch_file, 'w') as f:
            for pdf in batch:
                f.write(f"{pdf}\n")
        
        # Run worker subprocess
        cmd = [sys.executable, str(WORKER_SCRIPT), str(batch_file)]
        
        print(f"Starting worker subprocess...")
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=1800  # 30 minutes max per batch
            )
            
            elapsed = time.time() - start_time
            print(f"Worker completed in {elapsed:.1f}s")
            print()
            
            # Parse worker output
            if result.returncode == 0:
                # Extract results from output
                output = result.stdout
                
                # Count successes in this batch
                success_count = output.count("[SUCCESS]")
                
                # Update state
                for pdf in batch:
                    if pdf.name not in output or "[FAILED]" in output:
                        state["failed"].append(pdf.name)
                    else:
                        state["processed"].append(pdf.name)
                
                # Try to extract chunk count
                for line in output.split('\n'):
                    if "Total chunks:" in line:
                        try:
                            chunks = int(line.split(':')[1].strip().replace(',', ''))
                            state["total_chunks"] += chunks
                        except:
                            pass
                
                print(f"Batch {batch_num} completed: {success_count}/{len(batch)} PDFs successful")
                
            else:
                print(f"[WARNING] Worker exited with code {result.returncode}")
                print("Output:", result.stdout[:500])
                print("Error:", result.stderr[:500])
                
                # Mark all as failed
                for pdf in batch:
                    if pdf.name not in state["processed"]:
                        state["failed"].append(pdf.name)
            
            # Save state after each batch
            save_state(state)
            print(f"State saved. Total processed: {len(state['processed'])}/{len(pdf_files)}")
            print()
            
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Batch {batch_num} took too long (>30min)")
            for pdf in batch:
                if pdf.name not in state["processed"]:
                    state["failed"].append(pdf.name)
            save_state(state)
            print()
        
        except Exception as e:
            print(f"[ERROR] Batch {batch_num} failed: {e}")
            for pdf in batch:
                if pdf.name not in state["processed"]:
                    state["failed"].append(pdf.name)
            save_state(state)
            print()
        
        finally:
            # Cleanup batch file
            if batch_file.exists():
                batch_file.unlink()
        
        # Short delay between batches
        if batch_idx + BATCH_SIZE < len(remaining):
            print("Waiting 5 seconds before next batch...")
            time.sleep(5)
            print()
    
    # Final report
    print()
    print("=" * 80)
    print("INDEXING COMPLETE")
    print("=" * 80)
    
    state = load_state()  # Reload final state
    
    print(f"Total PDFs: {len(pdf_files)}")
    print(f"Successfully processed: {len(state['processed'])}")
    print(f"Failed: {len(state['failed'])}")
    print(f"Total chunks indexed: {state['total_chunks']:,}")
    
    if state["failed"]:
        print(f"\nFailed PDFs ({len(state['failed'])}):")
        for name in state["failed"][:10]:  # Show first 10
            print(f"  - {name}")
        if len(state["failed"]) > 10:
            print(f"  ... and {len(state['failed']) - 10} more")
    
    print(f"\nCollection: biostatistics_main")
    print(f"Path: {BASE_DIR / 'rag_system' / 'chroma_db'}")

if __name__ == "__main__":
    main()
