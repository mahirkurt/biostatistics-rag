"""
Biostatistics RAG Query Interface - Pure ONNX Edition
======================================================
Query biostatistics methodology knowledge base with hybrid retrieval
Uses Pure ONNX MiniLM (no PyTorch dependency!)

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

# Set encoding before any other imports
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Platform mocking for ChromaDB
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

import anthropic
import chromadb
import numpy as np
from dotenv import load_dotenv

# Add parent to path for relative imports when run directly
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
load_dotenv(override=True)

# Get API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()

# Configuration
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "biostatistics_main"
LLM_MODEL = "claude-sonnet-4-20250514"
LLM_MAX_TOKENS = 4096
LLM_TEMPERATURE = 0.3


class PureONNXRetriever:
    """Simple retriever using Pure ONNX embeddings"""
    
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve documents for a query"""
        # Embed query
        query_embedding = self.embedder.encode([query])[0].tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        if not results["documents"][0]:
            return []
        
        # Format results
        documents = []
        for i in range(len(results["documents"][0])):
            score = 1.0 - results["distances"][0][i]  # Convert distance to similarity
            documents.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": score
            })
        
        return documents
    
    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "name": self.collection.name,
            "count": self.collection.count()
        }


class BiostatisticsRAG:
    """
    Biostatistics RAG System - Pure ONNX Edition
    
    Features:
    - Pure ONNX MiniLM embeddings (no PyTorch!)
    - ChromaDB vector search
    - Claude Sonnet 4.5 for answer generation
    - Turkish biostatistics guidance
    """
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._embedder = None
        self._retriever = None
        self._anthropic_client = None
        
        if self.verbose:
            print("\n" + "=" * 70)
            print("BIOSTATISTICS RAG - Pure ONNX Edition")
            print("=" * 70)
    
    @property
    def embedder(self):
        """Lazy load embedder"""
        if self._embedder is None:
            if self.verbose:
                print("\n[*] Loading Pure ONNX MiniLM embedder...")
            
            # Import here to avoid loading at module import time
            from rag_system.embeddings.pure_onnx_minilm import PureONNXMiniLMAdapter
            self._embedder = PureONNXMiniLMAdapter()
            
            if self.verbose:
                print("[OK] Embedder loaded (384-dim, padded to 1024)")
        
        return self._embedder
    
    @property
    def retriever(self):
        """Lazy load retriever"""
        if self._retriever is None:
            if self.verbose:
                print("\n[*] Connecting to ChromaDB...")
            
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
            collection = client.get_collection(name=COLLECTION_NAME)
            
            self._retriever = PureONNXRetriever(collection, self.embedder)
            
            if self.verbose:
                stats = self._retriever.get_collection_stats()
                print(f"[OK] Collection '{stats['name']}' with {stats['count']} documents")
        
        return self._retriever
    
    @property
    def anthropic_client(self):
        """Lazy load Anthropic client"""
        if self._anthropic_client is None:
            self._anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        return self._anthropic_client
    
    def search_only(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Search documents without LLM generation
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of documents with metadata and scores
        """
        return self.retriever.retrieve(query=query, top_k=top_k)
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True  # Kept for API compatibility
    ) -> Dict:
        """
        Full RAG query with LLM answer generation
        
        Args:
            query: User question
            top_k: Number of source documents
            use_reranking: Ignored (no reranking in pure ONNX version)
            
        Returns:
            Dict with answer, sources, and metadata
        """
        if self.verbose:
            print(f"\n[QUERY] {query}")
        
        # Retrieve documents
        docs = self.retriever.retrieve(query=query, top_k=top_k)
        
        if not docs:
            return {
                "answer": "Üzgünüm, bu soruya cevap verebilecek ilgili bilgi bulamadım.",
                "sources": [],
                "query": query,
                "timestamp": datetime.now().isoformat()
            }
        
        # Format context
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            score = doc.get('score', 0.0)
            content = doc['content']
            context_parts.append(f"[Kaynak {i}: {source}] (Skor: {score:.2f})\n{content}")
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer
        answer = self._generate_answer(query, context, docs)
        
        return {
            "answer": answer,
            "sources": docs,
            "query": query,
            "model": LLM_MODEL,
            "timestamp": datetime.now().isoformat()
        }
    
    def _generate_answer(self, query: str, context: str, docs: List[Dict]) -> str:
        """Generate answer using Claude"""
        
        system_prompt = """Sen biyoistatistik alanında uzman bir danışmansın.
Görevin, kullanıcıların biyoistatistik ve klinik araştırma yöntemleri hakkındaki 
sorularına verilen kaynaklara dayanarak detaylı, doğru ve anlaşılır cevaplar vermek.

Uzmanlık alanların:
- İstatistiksel test seçimi (parametrik/non-parametrik)
- Güç analizi ve örneklem büyüklüğü hesaplama  
- Klinik araştırma tasarımı (RCT, kohort, kesitsel)
- Regresyon analizi (lineer, lojistik, Cox)
- Çoklu karşılaştırma düzeltmeleri
- Meta-analiz ve sistematik derleme
- Survival analizi
- ROC analizi ve tanı testleri

Cevaplarında:
- Verilen kaynaklardaki bilgilere sadık kal
- Türkçe ve net bir dil kullan
- İstatistiksel kavramları açıkla
- Uygun olduğunda örnekler ve formüller ver
- Kaynak numaralarını [Kaynak N] şeklinde referans göster
- Eğer kaynaklarda yeterli bilgi yoksa bunu belirt"""

        user_prompt = f"""Soru: {query}

Aşağıdaki kaynaklara dayanarak bu soruyu yanıtla:

{context}

---

Lütfen kapsamlı, yapılandırılmış ve detaylı bir cevap ver."""

        try:
            response = self.anthropic_client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"LLM yanıt oluştururken hata: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return self.retriever.get_collection_stats()


def interactive_mode():
    """Interactive query mode for testing"""
    print("\n" + "=" * 70)
    print("BIOSTATISTICS RAG - INTERACTIVE MODE")
    print("=" * 70)
    
    rag = BiostatisticsRAG(verbose=True)
    
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'stats' to see collection statistics")
    print("  - 'search <query>' to search without LLM")
    print("  - 'quit' or 'exit' to exit")
    print("=" * 70)
    
    while True:
        try:
            query_text = input("\n🔍 Query: ").strip()
            
            if not query_text:
                continue
            
            if query_text.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break
            
            if query_text.lower() == 'stats':
                stats = rag.get_stats()
                print(f"\n📊 Collection: {stats['name']}")
                print(f"   Documents: {stats['count']}")
                continue
            
            if query_text.lower().startswith('search '):
                search_query = query_text[7:].strip()
                results = rag.search_only(search_query, top_k=5)
                print(f"\n📚 Found {len(results)} results:")
                for i, doc in enumerate(results, 1):
                    print(f"\n[{i}] Score: {doc['score']:.3f}")
                    print(f"    Source: {doc['metadata'].get('source', 'Unknown')}")
                    print(f"    Content: {doc['content'][:200]}...")
                continue
            
            # Full RAG query
            result = rag.query(query_text, top_k=5)
            
            print("\n" + "=" * 70)
            print("ANSWER:")
            print("=" * 70)
            print(result["answer"])
            print("\n" + "=" * 70)
            print(f"SOURCES ({len(result['sources'])}):")
            print("=" * 70)
            for i, src in enumerate(result['sources'], 1):
                print(f"  [{i}] {src['metadata'].get('source', 'Unknown')} (score: {src['score']:.3f})")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    interactive_mode()
