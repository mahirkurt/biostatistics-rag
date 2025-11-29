"""Test if transformers/sentence-transformers can import on Python 3.13"""
import sys
print(f"Python version: {sys.version}")

print("\nTesting transformers import...")
try:
    import transformers
    print(f"[OK] transformers version: {transformers.__version__}")
except Exception as e:
    print(f"[FAILED] {e}")

print("\nTesting sentence-transformers import...")
try:
    import sentence_transformers
    print(f"[OK] sentence-transformers version: {sentence_transformers.__version__}")
except Exception as e:
    print(f"[FAILED] {e}")

print("\nTesting SentenceTransformer model creation...")
try:
    from sentence_transformers import SentenceTransformer
    print("[OK] SentenceTransformer imported")
    
    # Try to instantiate a small model
    print("Loading tiny model for test...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print(f"[OK] Model loaded: {model}")
except Exception as e:
    print(f"[FAILED] {e}")
    import traceback
    traceback.print_exc()
