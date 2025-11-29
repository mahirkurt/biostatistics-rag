"""
Biostatistics RAG Query Interface
==================================
Query biostatistics methodology knowledge base with hybrid retrieval
Now fully standalone - no external dependencies!

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import sys
from pathlib import Path
import anthropic

# Import local embeddings and retrievers
from .embeddings import BGEModel, NomicModel
from .retrievers import HybridRetriever

# Import config
from .config.biostat_rag_config import *


class BiostatisticsRAG:
    """Biostatistics RAG query interface"""

    def __init__(self):
        print("\n" + "=" * 80)
        print("BIOSTATISTICS RAG - QUERY INTERFACE")
        print("=" * 80)

        print("\n[*] Initializing models...")

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

        # Initialize LLM client
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        print("\n[OK] Biostatistics RAG ready!")
        print("=" * 80)

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True
    ):
        """
        Full RAG query with LLM answer generation

        Args:
            query: User question
            top_k: Number of source documents to retrieve
            use_reranking: Whether to use re-ranking

        Returns:
            Dict with answer and sources
        """
        # Retrieve relevant documents
        docs = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking
        )

        if not docs:
            return {
                "answer": "Üzgünüm, bu soruya cevap verebilecek ilgili bilgi bulamadım.",
                "sources": [],
                "query": query
            }

        # Prepare context from retrieved documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']
            context_parts.append(f"[Kaynak {i}: {source}]\n{content}")

        context = "\n\n".join(context_parts)

        # Generate answer with Claude
        system_prompt = """Sen biyoistatistik alanında uzman bir danışmansın. 
Görevin, kullanıcıların biyoistatistik ve klinik araştırma yöntemleri hakkındaki 
sorularına verilen kaynaklara dayanarak detaylı, doğru ve anlaşılır cevaplar vermek.

Cevaplarında:
- Verilen kaynaklardaki bilgilere sadık kal
- Türkçe ve net bir dil kullan
- İstatistiksel kavramları açıkla
- Uygun olduğunda örnekler ver
- Kaynakları referans göster
- Eğer kaynaklarda yeterli bilgi yoksa bunu belirt"""

        user_prompt = f"""Soru: {query}

Aşağıdaki kaynaklara dayanarak bu soruyu yanıtla:

{context}

Lütfen kapsamlı ve detaylı bir cevap ver."""

        try:
            response = self.anthropic_client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )

            answer = response.content[0].text

        except Exception as e:
            answer = f"LLM yanıt oluştururken hata: {str(e)}"

        return {
            "answer": answer,
            "sources": docs,
            "query": query
        }

    def search_only(
        self,
        query_text: str,
        top_k: int = 5,
        use_reranking: bool = True
    ):
        """
        Search documents without LLM generation

        Args:
            query_text: Query text
            top_k: Number of results
            use_reranking: Whether to use re-ranking

        Returns:
            List of documents
        """
        return self.retriever.retrieve(
            query=query_text,
            top_k=top_k,
            use_reranking=use_reranking
        )

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

            # Query with LLM
            result = rag.query(query_text)
            print("\n" + "=" * 80)
            print("ANSWER:")
            print("=" * 80)
            print(result['answer'])
            print("\n" + "=" * 80)
            print(f"Sources: {len(result['sources'])}")
            print("=" * 80)

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

    for idx, query_text in enumerate(test_queries_list, 1):
        print(f"\n[Test {idx}/{len(test_queries_list)}]")
        print(f"Query: {query_text}")
        print("-" * 80)

        result = rag.query(query_text)

        if result['sources']:
            print(f"✅ Retrieved {len(result['sources'])} sources")
            print(f"\nAnswer: {result['answer'][:200]}...")
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