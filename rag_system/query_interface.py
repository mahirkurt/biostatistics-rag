"""
Advanced Biostatistics RAG Query Interface
==========================================
Production-grade RAG system with:
- Multi-query retrieval
- Hybrid search with PubMedBERT reranking
- Claude Sonnet 4.5 integration
- Biostatistics domain-specific prompting
- Turkish language support
- Citation and source tracking
- **NEW: Semantic analysis with cross-source verification**
- **NEW: Evidence strength scoring**
- **NEW: Multi-perspective synthesis**

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-27
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import anthropic
from dotenv import load_dotenv

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))

# Lazy imports - models loaded only when needed
_embeddings_module = None
_retrievers_module = None
_reasoning_module = None


def _get_embeddings():
    """Lazy load embeddings module"""
    global _embeddings_module
    if _embeddings_module is None:
        from embedding_models import BGEModel, PubMedBERTModel
        _embeddings_module = {"BGEModel": BGEModel, "PubMedBERTModel": PubMedBERTModel}
    return _embeddings_module


def _get_retrievers():
    """Lazy load retrievers module"""
    global _retrievers_module
    if _retrievers_module is None:
        from retrievers import HybridRetriever
        _retrievers_module = {"HybridRetriever": HybridRetriever}
    return _retrievers_module


def _get_reasoning():
    """Lazy load advanced reasoning module"""
    global _reasoning_module
    if _reasoning_module is None:
        from advanced_reasoning import (
            SemanticAnalyzer, 
            MultiPerspectiveSynthesizer,
            create_enhanced_context
        )
        _reasoning_module = {
            "SemanticAnalyzer": SemanticAnalyzer,
            "MultiPerspectiveSynthesizer": MultiPerspectiveSynthesizer,
            "create_enhanced_context": create_enhanced_context
        }
    return _reasoning_module


from config.biostat_rag_config import *

# Load environment variables
load_dotenv(override=True)

# Clean API key
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "").strip()


class BiostatisticsRAG:
    """
    Advanced Biostatistics RAG System

    Features:
    - Multi-query generation for comprehensive retrieval
    - Hybrid retrieval (BGE-M3 + PubMedBERT reranking)
    - Claude Sonnet 4.5 for answer generation
    - Biostatistics-specific prompting (Turkish support)
    - Citation tracking and formatting
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

        if self.verbose:
            print("\n" + "=" * 80)
            print("BIOSTATISTICS RAG - ADVANCED QUERY INTERFACE")
            print("=" * 80)
            print("\nInitializing models...")

        # Lazy load modules
        embeddings = _get_embeddings()
        retrievers = _get_retrievers()

        BGEModel = embeddings["BGEModel"]
        PubMedBERTModel = embeddings["PubMedBERTModel"]
        HybridRetriever = retrievers["HybridRetriever"]

        # Initialize embedding models
        self.primary = BGEModel(PRIMARY_MODEL_NAME)
        self.secondary = PubMedBERTModel(SECONDARY_MODEL_NAME)

        # Initialize hybrid retriever
        self.retriever = HybridRetriever(
            primary_model=self.primary,
            secondary_model=self.secondary,
            chroma_db_path=str(CHROMA_DB_PATH),
            collection_name=COLLECTION_NAME,
            top_k_retrieve=TOP_K_RETRIEVE,
            top_k_final=TOP_K_FINAL,
        )

        # Initialize Claude
        self.anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # Initialize semantic analyzer (uses primary model for similarity)
        reasoning = _get_reasoning()
        self.semantic_analyzer = reasoning["SemanticAnalyzer"](
            embedding_model=self.primary,
            similarity_threshold=0.75
        )
        self.synthesizer = reasoning["MultiPerspectiveSynthesizer"](self.semantic_analyzer)

        if self.verbose:
            print("\nBiostatistics RAG ready!")
            print(f"   - Embedding: {PRIMARY_MODEL_NAME}")
            print(f"   - Reranker: {SECONDARY_MODEL_NAME}")
            print(f"   - LLM: {LLM_MODEL}")
            print(f"   - Semantic Analysis: Enabled")
            print("=" * 80)

    def generate_sub_queries(self, query: str) -> List[str]:
        """
        Generate multiple related queries for comprehensive retrieval

        Args:
            query: Original user query

        Returns:
            List of related queries including the original
        """
        sub_query_prompt = f"""You are an expert in biostatistics and research methodology.
Given this query about biostatistics: "{query}"

Generate 2-3 related sub-queries that would help retrieve comprehensive information.
Focus on different aspects: methods, assumptions, alternatives, practical applications.

Return ONLY the queries, one per line, without numbering or bullets.
Keep queries in the same language as the original query."""

        try:
            response = self.anthropic_client.messages.create(
                model=LLM_MODEL,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": sub_query_prompt}],
            )

            sub_queries = [
                q.strip()
                for q in response.content[0].text.strip().split("\n")
                if q.strip()
            ]

            # Always include original query first
            all_queries = [query] + sub_queries

            if self.verbose:
                print(f"\n📝 Generated {len(sub_queries)} sub-queries:")
                for i, sq in enumerate(sub_queries, 1):
                    print(f"   {i}. {sq}")

            return all_queries[:4]  # Max 4 total queries

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Sub-query generation failed: {e}")
            return [query]

    def retrieve_with_multi_query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_multi_query: bool = True,
    ) -> List[Dict]:
        """
        Retrieve documents using multi-query strategy

        Args:
            query: User query
            top_k: Number of documents to return
            use_reranking: Use PubMedBERT reranking
            use_multi_query: Generate and use sub-queries

        Returns:
            List of retrieved documents with metadata
        """
        if use_multi_query:
            queries = self.generate_sub_queries(query)
        else:
            queries = [query]

        # Retrieve for each query
        all_results = []
        seen_chunks = set()

        for q in queries:
            results = self.retriever.query(q, filter_metadata=None, verbose=False)

            # Deduplicate by chunk content
            for r in results:
                chunk_id = r["metadata"].get("chunk_id", r["content"][:100])
                if chunk_id not in seen_chunks:
                    seen_chunks.add(chunk_id)
                    all_results.append(r)

        # Sort by score and take top-k
        all_results.sort(key=lambda x: x["score"], reverse=True)

        if self.verbose:
            print(f"\n🔍 Retrieved {len(all_results[:top_k])} unique documents")

        return all_results[:top_k]

    def format_context(self, documents: List[Dict]) -> Tuple[str, List[Dict]]:
        """
        Format retrieved documents into context for LLM

        Args:
            documents: Retrieved documents with metadata

        Returns:
            Tuple of (formatted_context, source_list)
        """
        context_parts = []
        sources = []

        for idx, doc in enumerate(documents, 1):
            source_name = doc["metadata"].get("source", "Unknown")
            chunk_id = doc["metadata"].get("chunk_id", 0)
            score = doc.get("score", 0.0)

            context_parts.append(
                f"[Kaynak {idx}] {source_name} (İlgililik: {score:.2f})\n"
                f"{doc['content']}\n"
            )

            sources.append(
                {
                    "source_id": idx,
                    "source": source_name,
                    "chunk_id": chunk_id,
                    "score": score,
                    "content": doc["content"],
                }
            )

        context = "\n---\n\n".join(context_parts)
        return context, sources

    def generate_answer(
        self, 
        query: str, 
        context: str, 
        sources: List[Dict],
        analysis_context: str = "",
        synthesis: Dict = None
    ) -> Dict:
        """
        Generate comprehensive answer using Claude with semantic analysis

        Args:
            query: User query
            context: Formatted context from retrieved documents
            sources: Source metadata
            analysis_context: Additional semantic analysis context
            synthesis: Full synthesis metadata

        Returns:
            Dict with answer, sources, and analysis information
        """
        # Enhanced system prompt with analysis awareness
        system_prompt = """Sen deneyimli bir biyoistatistik uzmanısın. Uzmanlık alanların:

- İstatistiksel test seçimi (parametrik/non-parametrik)
- Örneklem büyüklüğü ve güç analizi
- Regresyon analizi (lineer, lojistik, Cox)
- ANOVA ve varyantları
- Survival analizi
- Meta-analiz ve sistematik derleme
- Klinik araştırma istatistikleri
- Epidemiyolojik yöntemler

## Yanıt Oluşturma Kuralları:

1. **Kaynak Tabanlı Yanıt**: Yanıtlarını SADECE verilen kaynaklara dayandır
2. **Kaynak Atıfı**: Her önemli iddia için [Kaynak N] notasyonunu kullan
3. **Çapraz Kaynak Doğrulaması**: Birden fazla kaynak aynı bilgiyi destekliyorsa bunu belirt
4. **Çelişki Farkındalığı**: Kaynaklar arasında farklı görüşler varsa bunları açıkça belirt
5. **Kanıt Gücü**: Bilginin ne kadar güvenilir olduğunu değerlendir
6. **Eksik Bilgi**: Kaynaklar yeterli bilgi içermiyorsa bunu açıkça belirt

## Dil ve Format:
- Yanıtlarını Türkçe olarak ver
- Markdown formatını kullan (başlıklar, listeler, tablolar)
- Karmaşık konuları bölümler halinde yapılandır"""

        # Build enhanced user prompt
        user_prompt = f"""## Soru
{query}

## Biyoistatistik Kitaplarından Bağlam

{context}

"""
        # Add semantic analysis if available
        if analysis_context:
            user_prompt += f"""
---

## 🔍 Semantik Analiz Özeti
{analysis_context}

---
"""

        user_prompt += """
## Yanıt Talimatları

Lütfen kapsamlı ve yapılandırılmış bir cevap ver:

1. **Doğrudan Yanıt**: Soruyu net ve açık şekilde yanıtla
2. **Kaynak Gösterimi**: Önemli noktalar için [Kaynak N] ile kaynak göster
3. **Pratik Uygulama**: Mümkünse örnekler, formüller veya R/Python kodu ekle
4. **Varsayımlar**: Test varsayımlarını ve kontrol yöntemlerini belirt
5. **Alternatifler**: Uygun olduğunda alternatif yöntemleri öner
6. **Çapraz Doğrulama**: Birden fazla kaynak aynı bilgiyi destekliyorsa "[Kaynak X, Y, Z tarafından desteklenmektedir]" şeklinde belirt

## Yanıt:"""

        try:
            response = self.anthropic_client.messages.create(
                model=LLM_MODEL,
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )

            answer = response.content[0].text

            result = {
                "answer": answer,
                "sources": sources,
                "query": query,
                "model": LLM_MODEL,
                "timestamp": datetime.now().isoformat(),
            }
            
            # Add analysis metadata if available
            if synthesis:
                result["analysis"] = {
                    "evidence_strength": synthesis.get("evidence_strength", {}),
                    "thematic_clusters": list(synthesis.get("thematic_clusters", {}).keys()),
                    "cross_source_agreement": synthesis.get("cross_source_analysis", {}).get("agreement_score", 0),
                    "contradictions_found": len(synthesis.get("contradictions", [])),
                }

            return result

        except Exception as e:
            raise Exception(f"Answer generation failed: {str(e)}")

    def query(
        self,
        query: str,
        top_k: int = 5,
        use_reranking: bool = True,
        use_multi_query: bool = True,
        use_semantic_analysis: bool = True,
        return_sources_only: bool = False,
    ) -> Dict:
        """
        Complete RAG query pipeline with semantic analysis

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_reranking: Use PubMedBERT reranking
            use_multi_query: Generate sub-queries for retrieval
            use_semantic_analysis: Enable cross-source verification and synthesis
            return_sources_only: Skip LLM, return only retrieved docs

        Returns:
            Dict with answer, sources, analysis, and metadata
        """
        if self.verbose:
            print(f"\n" + "=" * 80)
            print(f"SORGU: {query}")
            print("=" * 80)

        # Step 1: Retrieve documents
        documents = self.retrieve_with_multi_query(
            query=query,
            top_k=top_k,
            use_reranking=use_reranking,
            use_multi_query=use_multi_query,
        )

        if not documents:
            return {
                "answer": "Bu sorgu için biyoistatistik bilgi tabanında ilgili bilgi bulunamadı.",
                "sources": [],
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }

        # Step 2: Format basic context
        context, sources = self.format_context(documents)

        # Step 3: Perform semantic analysis (if enabled)
        analysis_context = ""
        synthesis = None
        
        if use_semantic_analysis:
            if self.verbose:
                print(f"\n🔬 Semantik analiz yapılıyor...")
            
            try:
                # Convert sources back to document format for analysis
                docs_for_analysis = [
                    {
                        "content": s["content"],
                        "metadata": {"source": s["source"], "chunk_id": s["chunk_id"]},
                        "score": s["score"]
                    }
                    for s in sources
                ]
                
                # Generate synthesis
                synthesis = self.synthesizer.synthesize(query, docs_for_analysis)
                analysis_context = self.synthesizer.format_for_llm(synthesis)
                
                if self.verbose:
                    ev = synthesis["evidence_strength"]
                    print(f"   📊 Kanıt gücü: {ev['overall_strength']}/100")
                    print(f"   📚 Benzersiz kaynak: {ev['unique_sources']}")
                    print(f"   🔗 Çapraz kaynak uyumu: {synthesis['cross_source_analysis']['agreement_score']:.1%}")
                    print(f"   ⚠️  Çelişki: {len(synthesis.get('contradictions', []))}")
                    
            except Exception as e:
                if self.verbose:
                    print(f"   ⚠️  Semantik analiz hatası: {e}")
                analysis_context = ""
                synthesis = None

        # If only sources requested, return early
        if return_sources_only:
            result = {
                "sources": sources,
                "query": query,
                "timestamp": datetime.now().isoformat(),
            }
            if synthesis:
                result["analysis"] = synthesis
            return result

        # Step 4: Generate answer with Claude
        if self.verbose:
            print(f"\n🤖 {LLM_MODEL} ile yanıt oluşturuluyor...")

        result = self.generate_answer(
            query, 
            context, 
            sources, 
            analysis_context=analysis_context,
            synthesis=synthesis
        )

        if self.verbose:
            print(f"\nYanıt oluşturuldu ({len(result['answer'])} karakter)")
            if result.get("analysis"):
                print(f"📈 Kanıt gücü: {result['analysis']['evidence_strength'].get('overall_strength', 'N/A')}/100")
            print("=" * 80)

        return result

    def analyze_sources(self, query: str, top_k: int = 10) -> Dict:
        """
        Analyze sources without generating an answer
        
        Useful for understanding evidence quality before asking questions
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with detailed source analysis
        """
        documents = self.retrieve_with_multi_query(
            query=query,
            top_k=top_k,
            use_reranking=True,
            use_multi_query=True,
        )
        
        if not documents:
            return {"error": "No documents found", "query": query}
        
        # Convert to analysis format
        docs_for_analysis = [
            {
                "content": d["content"],
                "metadata": d["metadata"],
                "score": d["score"]
            }
            for d in documents
        ]
        
        # Full synthesis
        synthesis = self.synthesizer.synthesize(query, docs_for_analysis)
        
        # Add source previews
        synthesis["source_previews"] = [
            {
                "source": d["metadata"].get("source", "Unknown"),
                "score": round(d["score"], 3),
                "preview": d["content"][:200] + "..."
            }
            for d in documents[:5]
        ]
        
        return synthesis

    def search_only(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search without LLM - returns only retrieved documents

        Args:
            query: User query
            top_k: Number of documents to retrieve

        Returns:
            List of retrieved documents
        """
        return self.retriever.retrieve(query, top_k=top_k, use_reranking=True)

    def get_stats(self) -> Dict:
        """Get collection statistics"""
        return self.retriever.get_collection_stats()


def interactive_mode():
    """Interactive query mode for testing"""
    print("\n" + "=" * 80)
    print("BIOSTATISTICS RAG - INTERACTIVE MODE")
    print("=" * 80)

    rag = BiostatisticsRAG(verbose=True)

    print("\nKomutlar:")
    print("  - Sorunuzu yazın")
    print("  - 'stats' ile koleksiyon istatistiklerini görün")
    print("  - 'quit' veya 'exit' ile çıkış yapın")
    print("=" * 80)

    while True:
        try:
            query_text = input("\n🔍 Sorgu: ").strip()

            if not query_text:
                continue

            if query_text.lower() in ["quit", "exit"]:
                print("\n👋 Görüşmek üzere!")
                break

            if query_text.lower() == "stats":
                stats = rag.get_stats()
                print("\n📊 Koleksiyon İstatistikleri:")
                for key, value in stats.items():
                    print(f"   {key}: {value}")
                continue

            # Execute query
            result = rag.query(query_text, top_k=5, use_multi_query=True)

            print("\n" + "=" * 80)
            print("YANIT:")
            print("=" * 80)
            print(result["answer"])
            print("\n" + "=" * 80)
            print(f"KAYNAKLAR ({len(result['sources'])}):")
            print("=" * 80)
            for src in result["sources"]:
                print(f"  [{src['source_id']}] {src['source']} (skor: {src['score']:.3f})")

        except KeyboardInterrupt:
            print("\n\n👋 Görüşmek üzere!")
            break
        except Exception as e:
            print(f"\n❌ Hata: {e}")


if __name__ == "__main__":
    interactive_mode()
