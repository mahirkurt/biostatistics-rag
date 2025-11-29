"""
Standalone Retriever Module
============================
Hybrid retrieval with PubMedBERT reranking for Biostatistics RAG

Features:
- Primary retrieval with BGE-M3
- Reranking with PubMedBERT
- Local ChromaDB support
- Optional ChromaDB Cloud support

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-25
"""

import os

# ChromaDB Windows fix - must be before chromadb import
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# Platform mocking for ChromaDB 1.3.5 on Windows
import platform
platform.system = lambda: "Windows"
platform.win32_ver = lambda: ("10", "10.0.19041", "", "Multiprocessor Free")

import chromadb
from typing import List, Dict, Optional
import numpy as np


class HybridRetriever:
    """
    Hybrid retriever with primary model and optional PubMedBERT reranking
    
    Two-stage retrieval:
    1. Primary stage: BGE-M3 retrieves top-k documents from ChromaDB
    2. Reranking stage: PubMedBERT reranks based on medical domain knowledge
    """

    def __init__(
        self,
        primary_model,
        secondary_model=None,
        chroma_db_path: str = None,
        collection_name: str = "biostatistics_main",
        top_k_retrieve: int = 50,
        top_k_final: int = 20,
        use_cloud: bool = False,
        cloud_api_key: str = None,
        cloud_tenant: str = None,
        cloud_database: str = "default",
    ):
        """
        Initialize hybrid retriever

        Args:
            primary_model: Primary embedding model (BGE-M3)
            secondary_model: Secondary model for reranking (PubMedBERT), optional
            chroma_db_path: Path to local ChromaDB (if not using cloud)
            collection_name: ChromaDB collection name
            top_k_retrieve: Initial retrieval count from ChromaDB
            top_k_final: Final count after reranking
            use_cloud: Use ChromaDB Cloud instead of local
            cloud_api_key: ChromaDB Cloud API key
            cloud_tenant: ChromaDB Cloud tenant ID
            cloud_database: ChromaDB Cloud database name
        """
        self.primary = primary_model
        self.secondary = secondary_model
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final
        self.collection_name = collection_name

        # Initialize ChromaDB client
        if use_cloud and cloud_api_key and cloud_tenant:
            print(f"[*] Connecting to ChromaDB Cloud (tenant: {cloud_tenant[:8]}...)")
            self.client = chromadb.CloudClient(
                api_key=cloud_api_key, 
                tenant=cloud_tenant, 
                database=cloud_database
            )
            print("✅ Connected to ChromaDB Cloud")
        else:
            # Local ChromaDB with persistence
            if not chroma_db_path:
                raise ValueError("chroma_db_path required for local ChromaDB")
            print(f"[*] Connecting to local ChromaDB: {chroma_db_path}")
            self.client = chromadb.PersistentClient(path=str(chroma_db_path))
            print("✅ Connected to local ChromaDB")

        # Get or create collection
        self.collection = self.client.get_collection(name=collection_name)
        print(f"   Collection: {collection_name} ({self.collection.count()} documents)")

    def query(
        self,
        query_text: str,
        filter_metadata: Optional[Dict] = None,
        verbose: bool = False,
    ) -> List[Dict]:
        """
        Query with hybrid retrieval (BGE-M3 + PubMedBERT reranking)

        Args:
            query_text: Query text
            filter_metadata: Optional metadata filter for ChromaDB
            verbose: Print verbose output

        Returns:
            List of results with metadata and scores
        """
        # Stage 1: Encode query with primary model (BGE-M3)
        query_embedding = self.primary.encode(query_text, normalize=True)

        if verbose:
            print(f"[Stage 1] BGE-M3 retrieval (top-{self.top_k_retrieve})")

        # Retrieve from ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=self.top_k_retrieve,
            where=filter_metadata,
        )

        if not results["documents"][0]:
            return []

        # Format results
        documents = []
        for i in range(len(results["documents"][0])):
            doc = {
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],  # Convert distance to similarity
            }
            documents.append(doc)

        if verbose:
            print(f"   Retrieved {len(documents)} documents")

        # Stage 2: Rerank with secondary model (PubMedBERT)
        if self.secondary and len(documents) > 0:
            if verbose:
                print(f"[Stage 2] PubMedBERT reranking (top-{self.top_k_final})")
            
            doc_texts = [d["content"] for d in documents]
            rerank_scores = self.secondary.rerank(query_text, doc_texts)

            # Update scores with reranking scores
            for i, score in enumerate(rerank_scores):
                documents[i]["score"] = float(score)

            # Sort by new scores and take top-k
            documents.sort(key=lambda x: x["score"], reverse=True)
            documents = documents[:self.top_k_final]
            
            if verbose:
                print(f"   Reranked to {len(documents)} documents")

        return documents

    def retrieve(
        self, 
        query: str, 
        top_k: int = 5, 
        use_reranking: bool = True
    ) -> List[Dict]:
        """
        Simple retrieve interface

        Args:
            query: Query text
            top_k: Number of results to return
            use_reranking: Use PubMedBERT reranking

        Returns:
            List of documents with metadata and scores
        """
        if not use_reranking:
            # Temporarily disable reranking
            original_secondary = self.secondary
            self.secondary = None
            results = self.query(query, verbose=False)
            self.secondary = original_secondary
            return results[:top_k]

        return self.query(query, verbose=False)[:top_k]

    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search without reranking (fast mode)

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of documents
        """
        return self.retrieve(query, top_k=top_k, use_reranking=False)

    def get_collection_stats(self) -> Dict:
        """Get collection statistics"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
        }

    def print_results(self, results: List[Dict], max_text_length: int = 300):
        """Print formatted results"""
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] Score: {result['score']:.4f}")
            print(f"    Source: {result['metadata'].get('source', 'Unknown')}")
            content = result["content"]
            if len(content) > max_text_length:
                content = content[:max_text_length] + "..."
            print(f"    Content: {content}")


class SimpleRetriever:
    """
    Simple retriever without reranking (for compatibility)
    
    Uses only BGE-M3 for single-stage retrieval
    """

    def __init__(
        self,
        embedder,
        chroma_db_path: str,
        collection_name: str = "biostatistics_main",
    ):
        """
        Initialize simple retriever

        Args:
            embedder: Embedding model (BGE-M3)
            chroma_db_path: Path to ChromaDB
            collection_name: Collection name
        """
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(chroma_db_path))
        self.collection = self.client.get_collection(name=collection_name)
        print(f"✅ SimpleRetriever ready ({self.collection.count()} docs)")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Simple search

        Args:
            query: Query text
            top_k: Number of results

        Returns:
            List of documents
        """
        query_embedding = self.embedder.encode(query, normalize=True)
        
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
        )

        documents = []
        for i in range(len(results["documents"][0])):
            documents.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - results["distances"][0][i],
            })

        return documents

    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            "name": self.collection.name,
            "count": self.collection.count(),
        }


def test_retriever():
    """Test retriever with existing ChromaDB"""
    import sys
    from pathlib import Path
    
    # Add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    
    from embeddings import BGEModel, PubMedBERTModel
    from config.biostat_rag_config import CHROMA_DB_PATH, COLLECTION_NAME
    
    print("\n" + "=" * 60)
    print("HYBRID RETRIEVER TEST")
    print("=" * 60)
    
    # Initialize models
    print("\n[1] Loading models...")
    primary = BGEModel()
    secondary = PubMedBERTModel()
    
    # Initialize retriever
    print("\n[2] Initializing retriever...")
    retriever = HybridRetriever(
        primary_model=primary,
        secondary_model=secondary,
        chroma_db_path=str(CHROMA_DB_PATH),
        collection_name=COLLECTION_NAME,
        top_k_retrieve=20,
        top_k_final=5,
    )
    
    # Test query
    print("\n[3] Testing query...")
    query = "Üç grup karşılaştırması için hangi istatistiksel test kullanılır?"
    
    results = retriever.query(query, verbose=True)
    
    print(f"\n[Results for: '{query}']")
    retriever.print_results(results, max_text_length=200)
    
    print("\n" + "=" * 60)
    print("✅ Retriever test completed!")
    print("=" * 60)


if __name__ == "__main__":
    test_retriever()
