import chromadb
import numpy as np
import streamlit as st
import pandas as pd


@st.cache_resource
def get_client():
    return chromadb.PersistentClient(path="./chroma_db")


def main():
    st.set_page_config(page_title="Chroma DB Viewer", layout="wide")
    st.title("Chroma DB Viewer")

    client = get_client()
    collections = client.list_collections()

    if not collections:
        st.warning("No collections found in ./chroma_db")
        return

    col_names = [c.name for c in collections]
    selected = st.sidebar.selectbox("Collection", col_names)
    collection = client.get_collection(selected)

    count = collection.count()
    st.sidebar.metric("Total documents", count)

    if count == 0:
        st.info(f"Collection '{selected}' is empty.")
        return

    # Pagination
    page_size = st.sidebar.slider("Page size", 5, 50, 10)
    total_pages = max(1, (count + page_size - 1) // page_size)
    page = st.sidebar.number_input("Page", 1, total_pages, 1)

    offset = (page - 1) * page_size
    data = collection.get(
        limit=page_size,
        offset=offset,
        include=["documents", "metadatas", "embeddings"],
    )

    embeddings = data.get("embeddings")
    has_embeddings = embeddings is not None and len(embeddings) > 0

    # Documents table
    st.subheader(f"Documents (page {page}/{total_pages})")
    rows = []
    for i, doc_id in enumerate(data["ids"]):
        row = {"id": doc_id}
        if data["documents"]:
            text = data["documents"][i]
            row["document"] = text[:200] + "..." if len(text) > 200 else text
        if data["metadatas"]:
            row["metadata"] = str(data["metadatas"][i])
        if has_embeddings:
            vec = embeddings[i]
            row["embedding_dims"] = len(vec)
            row["embedding_preview"] = str(vec[:5])
        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True)

    # Expand a single document
    st.subheader("Full document view")
    doc_idx = st.selectbox("Select document", range(len(data["ids"])),
                           format_func=lambda i: data["ids"][i])
    if data["documents"]:
        st.text_area("Content", data["documents"][doc_idx], height=300)
    if data["metadatas"]:
        st.json(data["metadatas"][doc_idx])
    if has_embeddings:
        vec = embeddings[doc_idx]
        st.write(f"**Embedding dimensions:** {len(vec)}")
        st.line_chart(pd.DataFrame(vec, columns=["value"]))

        # 2D projection of all embeddings on this page
        if len(embeddings) > 1:
            st.subheader("2D Embedding projection (PCA)")
            arr = np.array(embeddings)
            mean = arr.mean(axis=0)
            centered = arr - mean
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ vt[:2].T
            proj_df = pd.DataFrame(projected, columns=["PC1", "PC2"])
            proj_df["id"] = data["ids"]
            st.scatter_chart(proj_df, x="PC1", y="PC2")


if __name__ == "__main__":
    main()
