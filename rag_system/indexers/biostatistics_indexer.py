"""
Biostatistics PDF Indexer
==========================
Indexes biostatistics methodology textbooks into ChromaDB using BGE-M3 embeddings

@author: Dr. Mahir Kurt + Claude Code
@date: 2025-11-24
"""

import sys
from pathlib import Path

# Import local embeddings
sys.path.insert(0, str(Path(__file__).parent.parent))
from embeddings.bge_m3 import BGEModel
import chromadb
from chromadb.config import Settings
from PyPDF2 import PdfReader
from typing import List, Dict
import numpy as np
from datetime import datetime

# Import config
sys.path.insert(0, str(Path(__file__).parent.parent))
from config.biostat_rag_config import *


class BiostatisticsIndexer:
    """Index biostatistics PDFs with BGE-M3 embeddings"""

    def __init__(self):
        print("\n" + "=" * 80)
        print("BIOSTATISTICS PDF INDEXER")
        print("=" * 80)

        self.model = BGEModel(PRIMARY_MODEL_NAME)
        self.client = None
        self.collection = None
        self.stats = {
            "total_pdfs": 0,
            "indexed_pdfs": 0,
            "total_chunks": 0,
            "failed_pdfs": [],
            "start_time": datetime.now()
        }

    def initialize_chromadb(self, delete_existing: bool = True):
        """Initialize ChromaDB collection"""
        print(f"\n📊 Initializing ChromaDB...")

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DB_PATH),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Delete existing collection if requested
        if delete_existing:
            try:
                self.client.delete_collection(COLLECTION_NAME)
                print(f"   🗑️  Deleted existing collection: {COLLECTION_NAME}")
            except:
                pass

        # Create collection
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

        print(f"   ✅ ChromaDB collection ready: {COLLECTION_NAME}")
        print(f"   Path: {CHROMA_DB_PATH}")

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF using PyPDF2"""
        try:
            reader = PdfReader(str(pdf_path))
            text = ""

            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            return text.strip()

        except Exception as e:
            print(f"   ⚠️  Error reading PDF: {e}")
            return ""

    def chunk_text(self, text: str) -> List[str]:
        """
        Chunk text with semantic boundaries

        Breaks at sentence endings (. ? ! \\n\\n) when possible
        """
        if not text:
            return []

        chunks = []
        start = 0

        while start < len(text):
            end = start + CHUNK_SIZE

            # Try to break at sentence boundary
            if end < len(text) and SEMANTIC_CHUNKING:
                for delimiter in ['. ', '? ', '! ', '\n\n']:
                    last_delim = text[start:end].rfind(delimiter)
                    if last_delim != -1:
                        end = start + last_delim + len(delimiter)
                        break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start with overlap
            start = end - CHUNK_OVERLAP if end < len(text) else end

        return chunks

    def index_pdf(self, pdf_path: Path) -> Dict:
        """Index a single PDF file"""
        print(f"\n📄 {pdf_path.name}")

        # Extract text
        print(f"   [1/3] Extracting text...")
        text = self.extract_text_from_pdf(pdf_path)

        if not text:
            error = "No text extracted"
            print(f"   ❌ {error}")
            return {"success": False, "error": error, "chunks": 0}

        # Chunk text
        print(f"   [2/3] Chunking text...")
        chunks = self.chunk_text(text)

        if not chunks:
            error = "No chunks created"
            print(f"   ❌ {error}")
            return {"success": False, "error": error, "chunks": 0}

        print(f"   Created {len(chunks)} chunks")

        # Index chunks
        print(f"   [3/3] Indexing chunks...")

        for i, chunk in enumerate(chunks):
            try:
                # Get embedding
                embedding = self.model.encode(chunk)

                # Ensure list format for ChromaDB
                if isinstance(embedding, np.ndarray):
                    embedding = embedding.tolist()

                # Create unique ID
                doc_id = f"{pdf_path.stem}_chunk_{i}"

                # Metadata
                metadata = {
                    "source": pdf_path.name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "type": "biostatistics",
                    "indexed_at": datetime.now().isoformat()
                }

                # Add to collection
                self.collection.add(
                    ids=[doc_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[metadata]
                )

            except Exception as e:
                print(f"   ⚠️  Error indexing chunk {i}: {e}")
                continue

        print(f"   ✅ Indexed {len(chunks)} chunks")

        return {"success": True, "error": None, "chunks": len(chunks)}

    def index_directory(self) -> Dict:
        """Index all PDFs in textbooks directory"""
        pdf_files = list(TEXTBOOKS_DIR.glob("*.pdf"))
        self.stats["total_pdfs"] = len(pdf_files)

        if not pdf_files:
            print(f"\n❌ No PDF files found in {TEXTBOOKS_DIR}")
            print(f"   Please add PDF files to this directory and re-run.")
            return self.stats

        print(f"\n📚 Found {len(pdf_files)} PDF files")
        print(f"   Directory: {TEXTBOOKS_DIR}")
        print(f"   Chunk size: {CHUNK_SIZE} chars (overlap: {CHUNK_OVERLAP})")
        print(f"   Model: {PRIMARY_MODEL_NAME}")

        # Index each PDF
        for idx, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{idx}/{len(pdf_files)}]", end=" ")

            result = self.index_pdf(pdf_path)

            if result["success"]:
                self.stats["indexed_pdfs"] += 1
                self.stats["total_chunks"] += result["chunks"]
            else:
                self.stats["failed_pdfs"].append({
                    "file": pdf_path.name,
                    "error": result["error"]
                })

        # Print summary
        self._print_summary()

        return self.stats

    def _print_summary(self):
        """Print indexing summary"""
        duration = datetime.now() - self.stats["start_time"]

        print("\n" + "=" * 80)
        print("INDEXING SUMMARY")
        print("=" * 80)

        print(f"\n✅ Indexing complete!")
        print(f"   Total PDFs: {self.stats['total_pdfs']}")
        print(f"   Indexed successfully: {self.stats['indexed_pdfs']}")
        print(f"   Total chunks: {self.stats['total_chunks']:,}")
        print(f"   Duration: {duration}")

        if self.stats["failed_pdfs"]:
            print(f"\n⚠️  Failed PDFs ({len(self.stats['failed_pdfs'])}):")
            for failed in self.stats["failed_pdfs"]:
                print(f"   - {failed['file']}: {failed['error']}")

        print(f"\n📊 Collection: {COLLECTION_NAME}")
        print(f"   Documents: {self.collection.count():,}")
        print(f"   Path: {CHROMA_DB_PATH}")

        print("\n" + "=" * 80)


def main():
    """Run Biostatistics indexer"""
    indexer = BiostatisticsIndexer()
    indexer.initialize_chromadb(delete_existing=True)
    indexer.index_directory()


if __name__ == "__main__":
    main()
