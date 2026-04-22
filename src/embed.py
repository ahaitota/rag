import torch
from langchain_huggingface import HuggingFaceEmbeddings

def get_embedder(
    model_name: str = "microsoft/harrier-oss-v1-0.6b",
    device: str = "cpu",
) -> HuggingFaceEmbeddings:
    """LangChain Embeddings backed by a sentence-transformers model from HF."""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device, "trust_remote_code": True},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )

def embed(chunks, model_name="microsoft/harrier-oss-v1-0.6b", device="cpu"):
    """Embed a list of LangChain Documents. Returns (embedder, vectors)."""
    embedder = get_embedder(model_name=model_name, device=device)
    texts = [c.page_content for c in chunks]
    vectors = embedder.embed_documents(texts)
    return vectors
