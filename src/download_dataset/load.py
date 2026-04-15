from datasets import load_dataset

def load():
    ragbench_techqa = load_dataset("rungalileo/ragbench", "techqa", split="train")
    print(ragbench_techqa[0])
    return ragbench_techqa

if __name__ == "__main__":
    dataset = load()
    print(dataset[0])