"""Test BGEModel import in indexer context"""
# Mock platform functions first (same as indexer)
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import platform
def _mock_system():
    return "Windows"
def _mock_win32_ver():
    return ("10", "10.0.19041", "", "Multiprocessor Free")
platform.system = _mock_system
platform.win32_ver = _mock_win32_ver

# Add rag_system to path (same as indexer)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "rag_system"))

print("Testing BGEModel import...")
try:
    from embeddings.bge_m3 import BGEModel
    print("[OK] BGEModel imported")
    
    print("\nCreating BGEModel instance (BAAI/bge-m3)...")
    model = BGEModel("BAAI/bge-m3")
    print(f"[OK] Model created: {model}")
    
except Exception as e:
    print(f"[FAILED] {e}")
    import traceback
    traceback.print_exc()
