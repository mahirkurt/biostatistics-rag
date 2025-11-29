"""
Biostatistics RAG Query Interface
==================================
Query biostatistics methodology knowledge base with hybrid retrieval

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import sys
from pathlib import Path

# Add shared components to path
SHARED_COMPONENTS_PATH = Path("D:/Repositories/shared_rag_components")
if SHARED_COMPONENTS_PATH.exists():
    sys.path.insert(0, str(SHARED_COMPONENTS_PATH))
else:
    print(f"❌ Error: Shared components not found at {SHARED_COMPONENTS_PATH}")
    sys.exit(1)

from embeddings import BGEModel, NomicModel
from retrievers import HybridRetriever

# Import config
sys.path.insert(0, str(Path(__file__).parent))
from config.biostat_rag_config import *


class BiostatisticsRAG:
    """Biostatistics RAG query interface"""

    def __init__(self):
        print("\n" + "=" * 80)
        print("BIOSTATISTICS RAG - QUERY INTERFACE")
        print("=" * 80)

        print(f"\n🔄 Initializing models...")

        # Initialize models
        self.primary = BGEModel(PRIMARY_MODEL_NAME)
        self.secondary = NomicModel(SECONDARY_MODEL_NAME)

        # Initialize retriever
        self.retriever = HybridRetriever(
            primary_model=self.primary,
            secondary_model=self.secondary,
            chroma_db_path=str(CHROMA_DB_PATH),
            collection_name=COLLECTION_NAME,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_final=TOP_K_FINAL
        )

        print(f"\n✅ Biostatistics RAG ready!")
        print("=" * 80)

    def query(
        self,
        query_text: str,
        filter_metadata: dict = None,
        verbose: bool = True
    ):
        """
        Query biostatistics knowledge base

        Args:
            query_text: Query text
            filter_metadata: Optional metadata filter
            verbose: Print results

        Returns:
            List of results
        """
        results = self.retriever.query(
            query_text,
            filter_metadata=filter_metadata,
            verbose=verbose
        )

        if verbose and results:
            self.retriever.print_results(results, max_text_length=300)

        return results

    def get_stats(self):
        """Get collection statistics"""
        stats = self.retriever.get_collection_stats()

        print("\n" + "=" * 80)
        print("COLLECTION STATISTICS")
        print("=" * 80)

        for key, value in stats.items():
            print(f"  {key}: {value}")

        print("=" * 80)

        return stats


def interactive_mode():
    """Interactive query mode"""
    rag = BiostatisticsRAG()

    print("\n" + "=" * 80)
    print("INTERACTIVE MODE")
    print("=" * 80)
    print("\nCommands:")
    print("  - Type your question")
    print("  - 'stats' to see collection statistics")
    print("  - 'quit' or 'exit' to exit")
    print("=" * 80)

    while True:
        try:
            query_text = input("\n🔍 Query: ").strip()

            if not query_text:
                continue

            if query_text.lower() in ['quit', 'exit']:
                print("\n👋 Goodbye!")
                break

            if query_text.lower() == 'stats':
                rag.get_stats()
                continue

            # Query
            rag.query(query_text)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")


def test_queries():
    """Run test queries"""
    rag = BiostatisticsRAG()

    test_queries_list = [
        "Sample size calculation for logistic regression",
        "When to use Mann-Whitney vs t-test?",
        "Mixed models for longitudinal data",
        "Power analysis for survival analysis",
        "Assumptions of linear regression"
    ]

    print("\n" + "=" * 80)
    print("RUNNING TEST QUERIES")
    print("=" * 80)

    for idx, query in enumerate(test_queries_list, 1):
        print(f"\n[Test {idx}/{len(test_queries_list)}]")
        print(f"Query: {query}")
        print("-" * 80)

        results = rag.query(query, verbose=False)

        if results:
            print(f"✅ Retrieved {len(results)} results")
            print(f"   Top score: {results[0]['score']:.4f}")
            print(f"   Top source: {results[0]['metadata']['source']}")
        else:
            print("❌ No results")

    print("\n" + "=" * 80)
    print("TEST QUERIES COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_queries()
        else:
            # Single query from command line
            query_text = " ".join(sys.argv[1:])
            rag = BiostatisticsRAG()
            rag.query(query_text)
    else:
        # Interactive mode
        interactive_mode()
