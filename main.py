from dotenv import load_dotenv
load_dotenv()

from src.pipeline import build_pipeline

if __name__ == "__main__":
    # Switch camera_type to "webcam" any time to go back to the laptop camera —
    # nothing else in the pipeline changes.
    pipeline = build_pipeline(single_person_mode=True, camera_type="webcam")
    pipeline.run()