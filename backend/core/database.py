"""
Database connections for ChromaDB and SQLite.
"""
import chromadb
from sentence_transformers import SentenceTransformer
from .config import CHROMADB_PATH, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# CHROMADB CLIENT
# ---------------------------------------------------------------------------
_chroma_client = None
_collection = None
_embedding_model = None


def get_chroma_client():
    """Get or create ChromaDB client."""
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMADB_PATH)
    return _chroma_client


def get_collection():
    """Get the main document collection."""
    global _collection
    if _collection is None:
        client = get_chroma_client()
        _collection = client.get_or_create_collection(
            name="capex_docs",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


def get_embedding_model():
    """Get or load the embedding model."""
    global _embedding_model
    if _embedding_model is None:
        print("Loading embedding model...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        print(f"✓ Loaded {EMBEDDING_MODEL}")
    return _embedding_model


def embed_text(text: str) -> list[float]:
    """Embed a single text string."""
    model = get_embedding_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple text strings."""
    model = get_embedding_model()
    return model.encode(texts).tolist()


# ---------------------------------------------------------------------------
# COLLECTION STATS
# ---------------------------------------------------------------------------
def get_collection_stats() -> dict:
    """Get statistics about the ChromaDB collection."""
    collection = get_collection()
    count = collection.count()
    
    if count == 0:
        return {
            "total_documents": 0,
            "companies": {},
            "filing_types": {},
        }
    
    # Get all metadata to compute stats
    results = collection.get(include=["metadatas"], limit=count)
    
    companies = {}
    filing_types = {}
    
    for meta in results["metadatas"]:
        company = meta.get("company", "Unknown")
        ftype = meta.get("filing_type", "Unknown")
        
        companies[company] = companies.get(company, 0) + 1
        filing_types[ftype] = filing_types.get(ftype, 0) + 1
    
    return {
        "total_documents": count,
        "companies": companies,
        "filing_types": filing_types,
    }
