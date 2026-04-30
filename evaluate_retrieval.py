import json
import numpy as np
from langchain_chroma import Chroma
from embed import get_embedder


def chunk_overlap(retrieved_chunk: str, ground_truth_chunk: str, threshold=0.3) -> bool:
    """Check if retrieved chunk overlaps with ground truth using token overlap."""
    ret_tokens = set(retrieved_chunk.lower().split())
    gt_tokens = set(ground_truth_chunk.lower().split())
    if not gt_tokens:
        return False
    overlap = len(ret_tokens & gt_tokens) / len(gt_tokens)
    return overlap >= threshold


def evaluate_retrieval(
    goldens_path="datasets/applets_goldens_120.json",
    collection_name="applets",
    persist_directory="./chroma_db",
    model_name="microsoft/harrier-oss-v1-0.6b",
    k=5,
    threshold=0.3,
):
    embedder = get_embedder(model_name=model_name)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embedder,
        persist_directory=persist_directory,
    )

    with open(goldens_path) as f:
        goldens = json.load(f)

    recalls = []
    precisions = []

    for golden in goldens:
        query = golden["input"]
        gt_contexts = golden["context"]

        results = vectorstore.similarity_search(query, k=k)
        retrieved_chunks = [r.page_content for r in results]

        # For each ground-truth chunk, check if any retrieved chunk covers it
        hits = 0
        for gt_chunk in gt_contexts:
            if any(chunk_overlap(ret, gt_chunk, threshold) for ret in retrieved_chunks):
                hits += 1
        recall = hits / len(gt_contexts) if gt_contexts else 0

        # For each retrieved chunk, check if it matches any ground-truth chunk
        relevant_retrieved = 0
        for ret in retrieved_chunks:
            if any(chunk_overlap(ret, gt_chunk, threshold) for gt_chunk in gt_contexts):
                relevant_retrieved += 1
        precision = relevant_retrieved / len(retrieved_chunks) if retrieved_chunks else 0

        recalls.append(recall)
        precisions.append(precision)

    mean_recall = np.mean(recalls)
    mean_precision = np.mean(precisions)
    f1 = 2 * mean_recall * mean_precision / (mean_recall + mean_precision + 1e-9)

    print(f"Recall@{k}:    {mean_recall:.3f} (std: {np.std(recalls):.3f})")
    print(f"Precision@{k}: {mean_precision:.3f} (std: {np.std(precisions):.3f})")
    print(f"F1@{k}:        {f1:.3f}")
    print(f"\nPer-query recall distribution:")
    print(f"  100% recall: {sum(1 for r in recalls if r == 1.0)} / {len(recalls)}")
    print(f"  >50% recall: {sum(1 for r in recalls if r > 0.5)} / {len(recalls)}")
    print(f"  0% recall:   {sum(1 for r in recalls if r == 0.0)} / {len(recalls)}")

    return {"recall": mean_recall, "precision": mean_precision, "f1": f1}


if __name__ == "__main__":
    evaluate_retrieval()
