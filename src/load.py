from typing import Any, cast
from datasets import load_dataset
from langchain_core.documents import Document
    
def load():
    ragbench_techqa = load_dataset("rungalileo/ragbench", "techqa", split="train")
    documents = []
    for row in ragbench_techqa:
        row_dict = cast(dict[str, Any], row)
        for doc_text in row_dict['documents']:
            doc = Document(page_content=doc_text)
            documents.append(doc)
    return documents

if __name__ == "__main__":
    dataset = load()