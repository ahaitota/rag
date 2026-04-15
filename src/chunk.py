from transformers import AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_chunks(documents):
    model_name = "microsoft/harrier-oss-v1-0.6b"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    text_splitter_small = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=128,    
        chunk_overlap=32
    )

    chunked_documents = text_splitter_small.split_documents(documents)
    return chunked_documents