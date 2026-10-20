import faiss
import numpy as np

class FaissVectorStore:
    """Stores vectors and performs fast similarity search."""
    
    def __init__(self, embedding_dim: int):
        # IndexFlatIP uses Inner Product (Cosine Similarity for normalized vectors)
        self.index = faiss.IndexFlatIP(embedding_dim)

    def add(self, embeddings):
        self.index.add(np.array(embeddings).astype("float32"))

    def search(self, query_embedding, top_k=5):
        scores, indices = self.index.search(
            np.array([query_embedding]).astype("float32"), 
            top_k
        )
        # Returns a list of tuples: (chunk_index, similarity_score)
        return list(zip(indices[0], scores[0]))vectore_
