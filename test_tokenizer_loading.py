"""
Pre-load BGE-M3 tokenizer and model to diagnose loading issues
"""
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

print("Testing tokenizer loading with timeout protection...")

import signal
import sys

def timeout_handler(signum, frame):
    print("\n[TIMEOUT] Loading took too long - system issue detected")
    sys.exit(1)

# Set 5-minute timeout
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)

try:
    print("\n1. Loading tokenizer...")
    from transformers import AutoTokenizer
    
    print("   Calling from_pretrained...")
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    print(f"   [OK] Tokenizer loaded: {type(tokenizer)}")
    
    print("\n2. Testing tokenizer...")
    result = tokenizer("test text", return_tensors="pt")
    print(f"   [OK] Tokenization works: {result['input_ids'].shape}")
    
    print("\n[SUCCESS] Tokenizer is working properly")
    print("The KeyboardInterrupt is likely from system interference (antivirus/indexing)")
    
except KeyboardInterrupt:
    print("\n[INTERRUPTED] Manual KeyboardInterrupt or system issue")
    print("Recommendations:")
    print("  1. Disable antivirus temporarily")
    print("  2. Stop Windows Search/Indexing")
    print("  3. Close other Python processes")
    print("  4. Try running as Administrator")
    
except Exception as e:
    print(f"\n[ERROR] {e}")
    import traceback
    traceback.print_exc()

finally:
    signal.alarm(0)
