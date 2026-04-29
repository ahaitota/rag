from langchain_chroma import Chroma
from langchain_core.documents import Document

from embed import get_embedder, embed


def create_vectorstore(
    chunks: list[Document],
    vectors: list[list[float]],
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    model_name: str = "microsoft/harrier-oss-v1-0.6b",
) -> Chroma:
    """Index small chunks into Chroma using pre-computed vectors."""
    embedder = get_embedder(model_name=model_name)
    texts = [doc.page_content for doc in chunks]
    metadatas = [doc.metadata if doc.metadata else None for doc in chunks]
    ids = [str(i) for i in range(len(chunks))]

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_directory,
    )

    add_kwargs = {
        "ids": ids,
        "documents": texts,
        "embeddings": vectors,
    }
    if any(m is not None for m in metadatas):
        add_kwargs["metadatas"] = [m or {"_placeholder": ""} for m in metadatas]

    vectorstore._collection.add(**add_kwargs)
    return vectorstore


def retrieve_parent_chunks(
    query: str,
    big_chunks: list[Document],
    k: int = 5,
    persist_directory: str = "./chroma_db",
    collection_name: str = "default",
    model_name: str = "microsoft/harrier-oss-v1-0.6b",
) -> list[Document]:
    """Search small chunks, return their parent big chunks."""
    embedder = get_embedder(model_name=model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_directory,
    )
    results = vectorstore.similarity_search(query, k=k)

    parent_map = {b.metadata["parent_id"]: b for b in big_chunks}

    seen = set()
    parents = []
    for doc in results:
        pid = doc.metadata.get("parent_id")
        if pid and pid not in seen:
            seen.add(pid)
            parents.append(parent_map[pid])

    return parents
