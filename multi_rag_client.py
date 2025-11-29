"""
Multi-RAG Client for CF Project
Connects to Medical Writing and Biostatistics RAG microservices
"""
import requests
from typing import Dict, List, Optional
import anthropic
import os


class MultiRAGClient:
    """Client to access multiple RAG services"""
    
    def __init__(
        self,
        medical_writing_url: str = "http://localhost:8001",
        biostatistics_url: str = "http://localhost:8002",
        anthropic_api_key: Optional[str] = None
    ):
        self.medical_writing_url = medical_writing_url
        self.biostatistics_url = biostatistics_url
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.anthropic_client = anthropic.Anthropic(api_key=self.anthropic_api_key)
    
    def check_services(self) -> Dict[str, bool]:
        """Check which RAG services are available"""
        status = {}
        
        try:
            resp = requests.get(f"{self.medical_writing_url}/health", timeout=2)
            status["medical_writing"] = resp.status_code == 200
        except:
            status["medical_writing"] = False
        
        try:
            resp = requests.get(f"{self.biostatistics_url}/health", timeout=2)
            status["biostatistics"] = resp.status_code == 200
        except:
            status["biostatistics"] = False
        
        return status
    
    def query_medical_writing(
        self, 
        query: str, 
        top_k: int = 5,
        use_reranking: bool = True
    ) -> Dict:
        """Query medical writing RAG"""
        try:
            response = requests.post(
                f"{self.medical_writing_url}/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def query_biostatistics(
        self, 
        query: str, 
        top_k: int = 5,
        use_reranking: bool = True
    ) -> Dict:
        """Query biostatistics RAG"""
        try:
            response = requests.post(
                f"{self.biostatistics_url}/query",
                json={
                    "query": query,
                    "top_k": top_k,
                    "use_reranking": use_reranking
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def search_medical_writing(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search medical writing without LLM"""
        try:
            response = requests.post(
                f"{self.medical_writing_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            return []
    
    def search_biostatistics(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search biostatistics without LLM"""
        try:
            response = requests.post(
                f"{self.biostatistics_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            response.raise_for_status()
            return response.json()["results"]
        except Exception as e:
            return []
    
    def query_both_rags(
        self,
        query: str,
        top_k: int = 5
    ) -> Dict:
        """
        Query both RAGs and synthesize answer
        Useful for questions spanning both domains
        """
        # Get results from both RAGs
        med_results = self.search_medical_writing(query, top_k=top_k)
        stat_results = self.search_biostatistics(query, top_k=top_k)
        
        # Combine sources
        all_sources = []
        
        for doc in med_results:
            all_sources.append(f"[Medical Writing - {doc['source']}]\n{doc['content']}")
        
        for doc in stat_results:
            all_sources.append(f"[Biostatistics - {doc['source']}]\n{doc['content']}")
        
        if not all_sources:
            return {
                "answer": "No relevant information found in either RAG system.",
                "sources_medical": [],
                "sources_biostat": []
            }
        
        # Synthesize with Claude
        context = "\n\n---\n\n".join(all_sources[:10])  # Top 10 combined
        
        prompt = f"""Based on the following excerpts from medical writing and biostatistics textbooks, answer the question.

Context:
{context}

Question: {query}

Provide a comprehensive answer that integrates information from both medical writing and biostatistics perspectives when relevant."""

        message = self.anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            "answer": message.content[0].text,
            "sources_medical": med_results,
            "sources_biostat": stat_results,
            "query": query
        }


# Example usage for CF project
if __name__ == "__main__":
    client = MultiRAGClient()
    
    # Check services
    print("Checking RAG services...")
    status = client.check_services()
    print(f"Medical Writing: {'✓' if status['medical_writing'] else '✗'}")
    print(f"Biostatistics: {'✓' if status['biostatistics'] else '✗'}")
    
    # Example query
    if status['medical_writing']:
        print("\n" + "="*80)
        print("Example: Medical Writing Query")
        print("="*80)
        result = client.query_medical_writing(
            "How to write the methods section of a clinical trial paper?"
        )
        print(f"\nAnswer: {result.get('answer', 'Error')[:500]}...")
    
    if status['biostatistics']:
        print("\n" + "="*80)
        print("Example: Biostatistics Query")
        print("="*80)
        result = client.query_biostatistics(
            "Which statistical test should I use for comparing three independent groups?"
        )
        print(f"\nAnswer: {result.get('answer', 'Error')[:500]}...")
    
    if all(status.values()):
        print("\n" + "="*80)
        print("Example: Combined Query (Both RAGs)")
        print("="*80)
        result = client.query_both_rags(
            "How to report statistical results in clinical trial manuscript?"
        )
        print(f"\nAnswer: {result['answer'][:500]}...")
