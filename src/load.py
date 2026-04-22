from typing import Any, cast
from datasets import load_dataset
from langchain_core.documents import Document
    
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

if __name__ == "__main__":
    dataset = load("rungalileo/ragbench", "techqa", "train")