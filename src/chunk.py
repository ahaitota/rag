import uuid

from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_splitter(model_name, chunk_size, chunk_overlap):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def get_small_splitter(model_name):
    return _get_splitter(model_name, chunk_size=128, chunk_overlap=32)


def get_big_splitter(model_name):
    return _get_splitter(model_name, chunk_size=512, chunk_overlap=64)


def split_small_chunks(documents, model_name):
    splitter = get_small_splitter(model_name)
    return splitter.split_documents(documents)


def split_big_chunks(documents, model_name):
    splitter = get_big_splitter(model_name)
    return splitter.split_documents(documents)


def split_with_parents(documents, model_name):
    """Split documents into big and small chunks, linking small chunks to their parent big chunk."""
    big_splitter = get_big_splitter(model_name)
    small_splitter = get_small_splitter(model_name)

    big_chunks = big_splitter.split_documents(documents)

    for big in big_chunks:
        big.metadata["parent_id"] = str(uuid.uuid4())

    small_chunks = []
    for big in big_chunks:
        children = small_splitter.split_documents([big])
        for child in children:
            child.metadata["parent_id"] = big.metadata["parent_id"]
        small_chunks.extend(children)

    return big_chunks, small_chunks