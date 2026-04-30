import glob
import os
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig
import yaml


def generate_dataset(
    docs_glob="datasets/applets/*.docx",
    output_path="datasets/applets_goldens_default",
    max_contexts_per_document=3,
    max_goldens_per_context=2,
    chunk_size=1024,
    chunk_overlap=50,
):
    """Generate a synthetic golden dataset from documents.

    number of goldens: max_contexts_per_document × max_goldens_per_context × num_docs
    """
    load_dotenv()

    synthesizer = Synthesizer(cost_tracking=True)
    doc_paths = glob.glob(docs_glob)

    context_config = ContextConstructionConfig(
        max_contexts_per_document=max_contexts_per_document,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=doc_paths,
        include_expected_output=True,
        max_goldens_per_context=max_goldens_per_context,
        context_construction_config=context_config,
    )

    dataframe = synthesizer.to_pandas()
    dataframe.to_csv(output_path + ".csv", index=False)
    dataframe.to_json(output_path + ".json", orient="records", indent=2)
    with open(output_path + ".yaml", "w") as f:
        yaml.dump(dataframe.to_dict(orient="records"), f, default_flow_style=False)
    print(f"Generated {len(dataframe)} goldens")
    print(f"DeepEval synthesis cost: ${synthesizer.synthesis_cost:.6f}")
    return dataframe


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head())
