from dotenv import load_dotenv
load_dotenv()
from src.pipeline import build_pipeline
from src.pipeline import build_pipeline

if __name__ == "__main__":
    # single_person_mode=True matches sprint scope: detect + track ONE person.
    # Flip to False any time to see multi-person tracking already works.
    pipeline = build_pipeline(single_person_mode=True)
    pipeline.run()