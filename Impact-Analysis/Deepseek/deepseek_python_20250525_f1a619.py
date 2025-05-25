from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict

class CodeRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.embeddings = {}
        self.docs = []
        self.doc_metadata = []
    
    def add_document(self, text: str, metadata: Dict):
        """Add a document to the retrieval system"""
        doc_id = len(self.docs)
        self.docs.append(text)
        self.doc_metadata.append(metadata)
        
        # Chunk the document if needed (for the 500 token limit)
        chunks = self._chunk_text(text, 400)  # Conservative chunk size
        for chunk in chunks:
            embedding = self.model.encode(chunk)
            self.embeddings[doc_id] = self.embeddings.get(doc_id, []) + [embedding]
    
    def _chunk_text(self, text: str, max_tokens: int) -> List[str]:
        """Split text into chunks that fit within token limits"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0
        
        for word in words:
            # Simple token approximation (1 token ≈ 4 chars)
            word_tokens = len(word) // 4 + 1
            if current_count + word_tokens > max_tokens:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_count = word_tokens
            else:
                current_chunk.append(word)
                current_count += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant documents"""
        query_embedding = self.model.encode(query)
        scores = []
        
        for doc_id, doc_embeddings in self.embeddings.items():
            # Compare query to each chunk of the document
            chunk_scores = [np.dot(query_embedding, emb) / 
                          (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
                          for emb in doc_embeddings]
            max_score = max(chunk_scores) if chunk_scores else 0
            scores.append((doc_id, max_score))
        
        # Sort by score
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top results with metadata
        return [{
            'score': score,
            'text': self.docs[doc_id],
            'metadata': self.doc_metadata[doc_id]
        } for doc_id, score in scores[:top_k]]