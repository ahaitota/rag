import glob
from dotenv import load_dotenv
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import ContextConstructionConfig

load_dotenv()

synthesizer = Synthesizer()
doc_paths = glob.glob("datasets/applets/*.docx")

context_config = ContextConstructionConfig(
    max_contexts_per_document=5,
    chunk_size=512,
    chunk_overlap=0,
    context_quality_threshold=0.5,
    context_similarity_threshold=0.5,
)

goldens = synthesizer.generate_goldens_from_docs(
    document_paths=doc_paths,
    include_expected_output=True,
    max_goldens_per_context=2,
    context_construction_config=context_config,
)

dataframe = synthesizer.to_pandas()
dataframe.to_csv("datasets/applets_goldens.csv", index=False)
print(f"Generated {len(dataframe)} goldens")
print(dataframe.head())
