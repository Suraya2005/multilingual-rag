from sentence_transformers import SentenceTransformer
import torch

class Embedder:
    """Generates 1024-dim vectors for text using the multilingual BGE-M3 model."""
    
    def __init__(self, model_name="BAAI/bge-m3", force_cpu=False):
        self.model = SentenceTransformer(model_name)
        
        if force_cpu:
            self.device = "cpu"
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model.to(self.device)
        print(f"Model: {model_name}")
        print(f"Device: {self.device}")

    def embed_query(self, text: str):
        return self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            device=self.device
        )
    
    def embed_chunks(self, chunks):
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=16,
            show_progress_bar=True,
            device=self.device,
            convert_to_numpy=True
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        return chunks