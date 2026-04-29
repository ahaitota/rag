import os
from pathlib import Path
from typing import Any, cast

import docx2txt
from pypdf import PdfReader
from datasets import load_dataset
from langchain_core.documents import Document


def _read_file(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    if ext == ".docx":
        return docx2txt.process(filepath)
    if ext == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(filepath).pages)
    return Path(filepath).read_text(encoding="utf-8")


def load_files(directory: str) -> list[Document]:
    """Read all files in a directory and return them as langchain Documents."""
    documents = []
    for filename in sorted(os.listdir(directory)):
        filepath = str(Path(directory) / filename)
        if not os.path.isfile(filepath):
            continue
        try:
            text = _read_file(filepath)
        except Exception:
            continue
        if text and text.strip():
            print(text)
            documents.append(Document(
                page_content=text,
                metadata={"source": filepath, "filename": filename},
            ))
    return documents

    
def load(path, name, split) -> list[Document]:
    ragbench_techqa = load_dataset(path=path, name=name, split=split)
    seen = set()
    documents = []
    for row in ragbench_techqa:
        row_dict = cast(dict[str, Any], row)
        for doc_text in row_dict['documents']:
            if doc_text not in seen:
                seen.add(doc_text)
                doc = Document(page_content=doc_text)
                documents.append(doc)
    return documents


def load_queries(path, name, split) -> list[dict]:
    """Load questions with their gold documents and responses for evaluation.

    Returns a list of dicts with keys: question, documents, response.
    """
    dataset = load_dataset(path=path, name=name, split=split)
    queries = []
    for row in dataset:
        row_dict = cast(dict[str, Any], row)
        queries.append({
            "question": row_dict["question"],
            "documents": row_dict["documents"],
            "response": row_dict["response"],
        })
    return queries

if __name__ == "__main__":
    dataset = load("rungalileo/ragbench", "techqa", "train")