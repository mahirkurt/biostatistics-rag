"""
Advanced Reasoning Module for Biostatistics RAG
================================================
Multi-dimensional semantic analysis with cross-source verification

Features:
- Semantic clustering of retrieved documents
- Cross-source consistency verification
- Contradiction detection
- Evidence strength scoring
- Multi-perspective synthesis

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-27
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass, field
import re


@dataclass
class EvidenceCluster:
    """A cluster of semantically related evidence"""
    cluster_id: int
    theme: str
    documents: List[Dict] = field(default_factory=list)
    avg_score: float = 0.0
    source_diversity: int = 0  # Number of unique sources
    consistency_score: float = 1.0  # 0-1, higher = more consistent


@dataclass
class CrossSourceAnalysis:
    """Results of cross-source verification"""
    agreement_score: float  # 0-1
    contradictions: List[Dict] = field(default_factory=list)
    supporting_pairs: List[Tuple[int, int]] = field(default_factory=list)
    unique_insights: List[Dict] = field(default_factory=list)


class SemanticAnalyzer:
    """
    Multi-dimensional semantic analysis for RAG results
    
    Analyzes retrieved documents for:
    - Thematic clustering
    - Cross-source verification
    - Evidence quality assessment
    """

    def __init__(self, embedding_model=None, similarity_threshold: float = 0.75):
        """
        Initialize analyzer
        
        Args:
            embedding_model: Model for computing embeddings (optional)
            similarity_threshold: Threshold for semantic similarity clustering
        """
        self.embedding_model = embedding_model
        self.similarity_threshold = similarity_threshold
        
        # Biostatistics domain keywords for theme detection
        self.theme_keywords = {
            "test_selection": [
                "t-test", "anova", "chi-square", "mann-whitney", "kruskal-wallis",
                "wilcoxon", "fisher", "mcnemar", "parametrik", "non-parametrik"
            ],
            "sample_size": [
                "sample size", "power", "örneklem", "güç analizi", "n=", "alpha",
                "beta", "effect size", "etki büyüklüğü"
            ],
            "assumptions": [
                "normallik", "normality", "homogenity", "homojenite", "varyans",
                "bağımsızlık", "independence", "varsayım", "assumption"
            ],
            "regression": [
                "regresyon", "regression", "lineer", "lojistik", "cox", "hazard",
                "odds ratio", "coefficient", "katsayı", "r-squared"
            ],
            "descriptive": [
                "ortalama", "mean", "medyan", "median", "standart sapma", "std",
                "variance", "range", "percentile", "quartile"
            ],
            "correlation": [
                "korelasyon", "correlation", "pearson", "spearman", "kendall",
                "association", "ilişki"
            ],
            "survival": [
                "survival", "sağkalım", "kaplan-meier", "log-rank", "hazard",
                "censoring", "time-to-event"
            ],
            "meta_analysis": [
                "meta-analiz", "meta-analysis", "heterogeneity", "forest plot",
                "funnel plot", "publication bias", "effect size"
            ]
        }

    def detect_themes(self, text: str) -> List[str]:
        """
        Detect biostatistics themes in text
        
        Args:
            text: Document text
            
        Returns:
            List of detected theme names
        """
        text_lower = text.lower()
        detected = []
        
        for theme, keywords in self.theme_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    if theme not in detected:
                        detected.append(theme)
                    break
        
        return detected if detected else ["general"]

    def cluster_by_theme(self, documents: List[Dict]) -> Dict[str, EvidenceCluster]:
        """
        Cluster documents by thematic similarity
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Dictionary of theme -> EvidenceCluster
        """
        clusters = defaultdict(lambda: EvidenceCluster(
            cluster_id=0, theme="", documents=[], avg_score=0.0
        ))
        
        for doc in documents:
            content = doc.get("content", "")
            themes = self.detect_themes(content)
            
            for theme in themes:
                if clusters[theme].theme == "":
                    clusters[theme].theme = theme
                    clusters[theme].cluster_id = len(clusters)
                
                clusters[theme].documents.append(doc)
        
        # Calculate cluster statistics
        for theme, cluster in clusters.items():
            if cluster.documents:
                # Average score
                scores = [d.get("score", 0) for d in cluster.documents]
                cluster.avg_score = sum(scores) / len(scores)
                
                # Source diversity
                unique_sources = set(
                    d.get("metadata", {}).get("source", "unknown") 
                    for d in cluster.documents
                )
                cluster.source_diversity = len(unique_sources)
        
        return dict(clusters)

    def compute_document_similarity(
        self, 
        doc1: Dict, 
        doc2: Dict
    ) -> float:
        """
        Compute semantic similarity between two documents
        
        Args:
            doc1, doc2: Document dictionaries with 'content' key
            
        Returns:
            Similarity score (0-1)
        """
        if self.embedding_model is None:
            # Fallback to keyword-based similarity
            return self._keyword_similarity(
                doc1.get("content", ""),
                doc2.get("content", "")
            )
        
        # Use embeddings
        emb1 = self.embedding_model.encode(doc1.get("content", ""))
        emb2 = self.embedding_model.encode(doc2.get("content", ""))
        
        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (
            np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8
        )
        
        return float(similarity)

    def _keyword_similarity(self, text1: str, text2: str) -> float:
        """Simple keyword-based similarity fallback"""
        words1 = set(re.findall(r'\b\w+\b', text1.lower()))
        words2 = set(re.findall(r'\b\w+\b', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

    def verify_cross_source_consistency(
        self, 
        documents: List[Dict]
    ) -> CrossSourceAnalysis:
        """
        Verify consistency across different sources
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            CrossSourceAnalysis with agreement score and details
        """
        analysis = CrossSourceAnalysis(agreement_score=1.0)
        
        if len(documents) < 2:
            return analysis
        
        # Group by source
        by_source = defaultdict(list)
        for doc in documents:
            source = doc.get("metadata", {}).get("source", "unknown")
            by_source[source].append(doc)
        
        # Find supporting pairs (similar content from different sources)
        sources = list(by_source.keys())
        support_count = 0
        total_pairs = 0
        
        for i in range(len(sources)):
            for j in range(i + 1, len(sources)):
                for doc1 in by_source[sources[i]]:
                    for doc2 in by_source[sources[j]]:
                        similarity = self.compute_document_similarity(doc1, doc2)
                        total_pairs += 1
                        
                        if similarity > self.similarity_threshold:
                            support_count += 1
                            analysis.supporting_pairs.append((
                                doc1.get("metadata", {}).get("chunk_id", 0),
                                doc2.get("metadata", {}).get("chunk_id", 0)
                            ))
        
        # Calculate agreement score
        if total_pairs > 0:
            analysis.agreement_score = support_count / total_pairs
        
        # Find unique insights (content with low similarity to others)
        for doc in documents:
            avg_sim = 0
            for other in documents:
                if doc != other:
                    avg_sim += self.compute_document_similarity(doc, other)
            
            if len(documents) > 1:
                avg_sim /= (len(documents) - 1)
            
            if avg_sim < 0.3:  # Low similarity = unique insight
                analysis.unique_insights.append({
                    "content": doc.get("content", "")[:200],
                    "source": doc.get("metadata", {}).get("source", "unknown"),
                    "uniqueness_score": 1 - avg_sim
                })
        
        return analysis

    def detect_contradictions(
        self, 
        documents: List[Dict]
    ) -> List[Dict]:
        """
        Detect potential contradictions in retrieved documents
        
        Looks for opposing statements about:
        - Statistical test recommendations
        - P-value interpretations
        - Effect size guidelines
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            List of potential contradiction pairs
        """
        contradictions = []
        
        # Contradiction patterns in biostatistics
        opposing_patterns = [
            # Test recommendations
            (r"use\s+(?:the\s+)?(\w+)\s+test", r"do\s+not\s+use\s+(?:the\s+)?(\w+)\s+test"),
            (r"(\w+)\s+(?:is|are)\s+(?:recommended|preferred)", r"(\w+)\s+(?:is|are)\s+(?:not\s+recommended|inappropriate)"),
            
            # Significance
            (r"p\s*[<≤]\s*0\.05\s+(?:is\s+)?significant", r"p\s*[<≤]\s*0\.05\s+(?:is\s+)?(?:not\s+)?(?:sufficient|meaningful)"),
            
            # Sample size
            (r"minimum\s+(?:sample\s+)?(?:size\s+)?(?:of\s+)?(\d+)", r"(?:at\s+least|minimum)\s+(?:of\s+)?(\d+)"),
        ]
        
        for i, doc1 in enumerate(documents):
            content1 = doc1.get("content", "").lower()
            
            for j, doc2 in enumerate(documents[i+1:], i+1):
                content2 = doc2.get("content", "").lower()
                
                # Check if from different sources
                source1 = doc1.get("metadata", {}).get("source", "")
                source2 = doc2.get("metadata", {}).get("source", "")
                
                if source1 == source2:
                    continue
                
                # Look for opposing statements
                for pattern1, pattern2 in opposing_patterns:
                    match1 = re.search(pattern1, content1)
                    match2 = re.search(pattern2, content2)
                    
                    if match1 and match2:
                        contradictions.append({
                            "doc1_source": source1,
                            "doc1_excerpt": content1[:200],
                            "doc2_source": source2,
                            "doc2_excerpt": content2[:200],
                            "type": "potential_contradiction"
                        })
                        break
        
        return contradictions

    def score_evidence_strength(self, documents: List[Dict]) -> Dict:
        """
        Score overall evidence strength
        
        Considers:
        - Number of sources
        - Source diversity
        - Cross-source agreement
        - Recency (if available)
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Evidence strength analysis
        """
        if not documents:
            return {
                "overall_strength": 0,
                "source_count": 0,
                "unique_sources": 0,
                "avg_relevance": 0,
                "recommendation": "Yetersiz kanıt"
            }
        
        # Count unique sources
        unique_sources = set(
            d.get("metadata", {}).get("source", "unknown") 
            for d in documents
        )
        
        # Average relevance score
        scores = [d.get("score", 0) for d in documents]
        avg_score = sum(scores) / len(scores)
        
        # Cross-source verification
        cross_analysis = self.verify_cross_source_consistency(documents)
        
        # Calculate overall strength (0-100)
        strength = 0
        
        # Source diversity (0-30 points)
        diversity_score = min(len(unique_sources) * 10, 30)
        strength += diversity_score
        
        # Relevance (0-30 points)
        relevance_score = avg_score * 30
        strength += relevance_score
        
        # Agreement (0-20 points)
        agreement_score = cross_analysis.agreement_score * 20
        strength += agreement_score
        
        # Volume (0-20 points)
        volume_score = min(len(documents) * 4, 20)
        strength += volume_score
        
        # Recommendation based on strength
        if strength >= 80:
            recommendation = "Çok güçlü kanıt - Güvenilir sonuçlar"
        elif strength >= 60:
            recommendation = "Güçlü kanıt - Genel güvenilirlik yüksek"
        elif strength >= 40:
            recommendation = "Orta düzey kanıt - Ek doğrulama önerilir"
        elif strength >= 20:
            recommendation = "Zayıf kanıt - Dikkatli yorumlanmalı"
        else:
            recommendation = "Yetersiz kanıt - Ek kaynaklara başvurun"
        
        return {
            "overall_strength": round(strength, 1),
            "source_count": len(documents),
            "unique_sources": len(unique_sources),
            "avg_relevance": round(avg_score, 3),
            "cross_source_agreement": round(cross_analysis.agreement_score, 3),
            "recommendation": recommendation,
            "unique_insights_count": len(cross_analysis.unique_insights)
        }


class MultiPerspectiveSynthesizer:
    """
    Synthesize information from multiple perspectives
    
    Creates structured summaries that:
    - Highlight consensus views
    - Note alternative approaches
    - Identify knowledge gaps
    """

    def __init__(self, analyzer: SemanticAnalyzer = None):
        self.analyzer = analyzer or SemanticAnalyzer()

    def synthesize(
        self, 
        query: str,
        documents: List[Dict]
    ) -> Dict:
        """
        Create multi-perspective synthesis of documents
        
        Args:
            query: Original user query
            documents: Retrieved documents
            
        Returns:
            Synthesis with multiple perspectives
        """
        # Cluster by theme
        clusters = self.analyzer.cluster_by_theme(documents)
        
        # Analyze evidence strength
        evidence = self.analyzer.score_evidence_strength(documents)
        
        # Cross-source analysis
        cross_analysis = self.analyzer.verify_cross_source_consistency(documents)
        
        # Detect contradictions
        contradictions = self.analyzer.detect_contradictions(documents)
        
        # Build synthesis
        synthesis = {
            "query": query,
            "evidence_strength": evidence,
            "thematic_clusters": {},
            "cross_source_analysis": {
                "agreement_score": cross_analysis.agreement_score,
                "supporting_pairs": len(cross_analysis.supporting_pairs),
                "unique_insights": cross_analysis.unique_insights[:3]  # Top 3
            },
            "contradictions": contradictions[:3] if contradictions else [],
            "perspectives": []
        }
        
        # Add thematic cluster summaries
        for theme, cluster in clusters.items():
            synthesis["thematic_clusters"][theme] = {
                "document_count": len(cluster.documents),
                "avg_score": round(cluster.avg_score, 3),
                "source_diversity": cluster.source_diversity,
                "top_sources": list(set(
                    d.get("metadata", {}).get("source", "")[:50] 
                    for d in cluster.documents[:3]
                ))
            }
        
        # Generate perspectives
        if "test_selection" in clusters:
            synthesis["perspectives"].append({
                "aspect": "Test Seçimi",
                "summary": self._summarize_cluster(clusters["test_selection"])
            })
        
        if "assumptions" in clusters:
            synthesis["perspectives"].append({
                "aspect": "Varsayımlar",
                "summary": self._summarize_cluster(clusters["assumptions"])
            })
        
        if "sample_size" in clusters:
            synthesis["perspectives"].append({
                "aspect": "Örneklem Büyüklüğü",
                "summary": self._summarize_cluster(clusters["sample_size"])
            })
        
        return synthesis

    def _summarize_cluster(self, cluster: EvidenceCluster) -> str:
        """Generate brief summary of a cluster"""
        if not cluster.documents:
            return "Bilgi bulunamadı"
        
        sources = set(
            d.get("metadata", {}).get("source", "")[:30] 
            for d in cluster.documents
        )
        
        return f"{len(cluster.documents)} belgeden, {len(sources)} farklı kaynaktan bilgi (ortalama skor: {cluster.avg_score:.2f})"

    def format_for_llm(self, synthesis: Dict) -> str:
        """
        Format synthesis as additional context for LLM
        
        Args:
            synthesis: Synthesis dictionary
            
        Returns:
            Formatted string for LLM context
        """
        parts = []
        
        # Evidence strength
        ev = synthesis["evidence_strength"]
        parts.append(f"""
## Kanıt Gücü Analizi
- Genel güç skoru: {ev['overall_strength']}/100
- Kaynak sayısı: {ev['source_count']} belge, {ev['unique_sources']} benzersiz kaynak
- Çapraz kaynak uyumu: {synthesis['cross_source_analysis']['agreement_score']:.1%}
- Değerlendirme: {ev['recommendation']}
""")
        
        # Thematic breakdown
        if synthesis["thematic_clusters"]:
            parts.append("\n## Tematik Dağılım")
            for theme, info in synthesis["thematic_clusters"].items():
                parts.append(f"- **{theme}**: {info['document_count']} belge, {info['source_diversity']} kaynak")
        
        # Contradictions warning
        if synthesis["contradictions"]:
            parts.append(f"\n⚠️ **Dikkat**: {len(synthesis['contradictions'])} potansiyel çelişki tespit edildi. Kaynaklar arasında farklı görüşler olabilir.")
        
        # Unique insights
        if synthesis["cross_source_analysis"]["unique_insights"]:
            parts.append("\n## Benzersiz Bulgular")
            for insight in synthesis["cross_source_analysis"]["unique_insights"]:
                parts.append(f"- [{insight['source'][:30]}]: {insight['content'][:100]}...")
        
        return "\n".join(parts)


def create_enhanced_context(
    query: str,
    documents: List[Dict],
    embedding_model=None
) -> Tuple[str, Dict]:
    """
    Create enhanced context with semantic analysis
    
    Args:
        query: User query
        documents: Retrieved documents
        embedding_model: Optional embedding model for similarity
        
    Returns:
        Tuple of (enhanced_context, analysis_metadata)
    """
    analyzer = SemanticAnalyzer(embedding_model)
    synthesizer = MultiPerspectiveSynthesizer(analyzer)
    
    # Generate synthesis
    synthesis = synthesizer.synthesize(query, documents)
    
    # Format analysis for LLM
    analysis_context = synthesizer.format_for_llm(synthesis)
    
    return analysis_context, synthesis


# Test function
if __name__ == "__main__":
    # Sample documents for testing
    test_docs = [
        {
            "content": "The t-test is used for comparing two groups. It assumes normal distribution.",
            "metadata": {"source": "Book1.pdf", "chunk_id": 1},
            "score": 0.9
        },
        {
            "content": "For non-normal data, use Mann-Whitney U test instead of t-test.",
            "metadata": {"source": "Book2.pdf", "chunk_id": 2},
            "score": 0.85
        },
        {
            "content": "Sample size calculation requires knowing expected effect size and alpha level.",
            "metadata": {"source": "Book1.pdf", "chunk_id": 3},
            "score": 0.8
        }
    ]
    
    analyzer = SemanticAnalyzer()
    synthesizer = MultiPerspectiveSynthesizer(analyzer)
    
    synthesis = synthesizer.synthesize("Which test should I use?", test_docs)
    
    print("\n" + "=" * 60)
    print("SYNTHESIS TEST")
    print("=" * 60)
    print(f"Evidence Strength: {synthesis['evidence_strength']['overall_strength']}/100")
    print(f"Themes: {list(synthesis['thematic_clusters'].keys())}")
    print(f"Recommendation: {synthesis['evidence_strength']['recommendation']}")
