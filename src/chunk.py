from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_small_chunks(documents, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter_small = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=128,    
        chunk_overlap=32
    )

    small_documents = text_splitter_small.split_documents(documents)
    return small_documents