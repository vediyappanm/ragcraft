from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_core.documents import Document

class ProductionRetriever:
    """Hybrid retrieval with re-ranking - current best practice."""
    
    def __init__(self, vector_store, documents: List[Document] = None):
        self.vector_store = vector_store
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Initialize BM25 if documents are provided
        if documents:
            self.bm25 = self._initialize_bm25(documents)
            self.documents = documents
        else:
            self.bm25 = None
            self.documents = None
        
    def _initialize_bm25(self, documents: List[Document]) -> BM25Okapi:
        """Initialize BM25 with document corpus."""
        tokenized_docs = []
        for doc in documents:
            # Simple tokenization - in production, use more sophisticated preprocessing
            tokens = doc.page_content.lower().split()
            tokenized_docs.append(tokens)
        
        return BM25Okapi(tokenized_docs)
        
    def retrieve(self, query: str, top_k: int = 10) -> List[Document]:
        """Hybrid retrieval combining vector search, BM25, and re-ranking."""
        if not self.documents:
            # Fallback to vector search only
            return self.vector_store.similarity_search(query, k=top_k)
        
        # Step 1: Get candidates from multiple sources
        vector_results = self.vector_store.similarity_search(query, k=top_k*2)
        bm25_results = self._bm25_search(query, top_k*2)
        
        # Step 2: Fuse results using Reciprocal Rank Fusion
        fused_results = self.reciprocal_rank_fusion(
            vector_results, 
            bm25_results
        )
        
        # Step 3: Re-rank with cross-encoder
        reranked = self.rerank_with_cross_encoder(query, fused_results[:top_k*3])
        
        return reranked[:top_k]
    
    def _bm25_search(self, query: str, top_k: int) -> List[Document]:
        """Search using BM25."""
        if not self.bm25:
            return []
        
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k documents
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            if bm25_scores[idx] > 0:  # Only include documents with positive scores
                doc = self.documents[idx]
                doc.metadata['bm25_score'] = float(bm25_scores[idx])
                results.append(doc)
        
        return results
    
    def reciprocal_rank_fusion(self, *result_lists, k: int = 60) -> List[Document]:
        """Standard Reciprocal Rank Fusion implementation."""
        fused_scores = {}
        doc_lookup = {}
        
        for rank, result_list in enumerate(result_lists):
            for position, doc in enumerate(result_list):
                doc_id = id(doc)  # Use object id as unique identifier
                
                if doc_id not in fused_scores:
                    fused_scores[doc_id] = 0
                    doc_lookup[doc_id] = doc
                
                # RRF formula: 1 / (k + position)
                fused_scores[doc_id] += 1.0 / (k + position + 1)
        
        # Sort by fused scores
        sorted_docs = sorted(
            fused_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return [doc_lookup[doc_id] for doc_id, _ in sorted_docs]
    
    def rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Document]:
        """Rerank candidates using a cross-encoder model."""
        if not documents:
            return []
        
        # Prepare input pairs for cross-encoder
        pairs = [[query, doc.page_content] for doc in documents]
        
        # Get relevance scores
        scores = self.reranker.predict(pairs)
        
        # Add scores to metadata and sort
        for doc, score in zip(documents, scores):
            doc.metadata['rerank_score'] = float(score)
        
        # Sort by rerank score
        reranked_docs = sorted(
            documents, 
            key=lambda x: x.metadata.get('rerank_score', 0), 
            reverse=True
        )
        
        return reranked_docs
