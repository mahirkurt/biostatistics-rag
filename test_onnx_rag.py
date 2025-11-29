"""Quick test for Pure ONNX RAG"""
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import platform
platform.system = lambda: 'Windows'
platform.win32_ver = lambda: ('10','10.0.19041','','')

import sys
sys.path.insert(0, '.')

from rag_system.query_interface_onnx import BiostatisticsRAG

print("Testing Pure ONNX Biostatistics RAG...")
print("=" * 50)

rag = BiostatisticsRAG(verbose=True)

print("\n[TEST 1] Search only...")
results = rag.search_only('power analysis sample size', top_k=3)
print(f"Found {len(results)} results:")
for i, r in enumerate(results, 1):
    source = r["metadata"].get("source", "Unknown")
    print(f"  [{i}] Score: {r['score']:.3f} - {source[:50]}")

print("\n[TEST 2] Stats...")
stats = rag.get_stats()
print(f"Collection: {stats['name']}, Documents: {stats['count']}")

print("\n" + "=" * 50)
print("SUCCESS! Pure ONNX RAG is working.")
