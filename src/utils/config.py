import os
from pathlib import Path

ROOT = os.environ.get("ROOT_DIR", Path(os.path.abspath(__file__)).parents[2])
CHECKPOINTS = os.environ.get("CHECKPOINTS_DIR", os.path.join(ROOT, "checkpoints"))
DATA = os.environ.get("DATA_DIR", os.path.join(ROOT, "data"))

RESULTS = os.environ.get("RESULTS_DIR", "results/")
IMAGES = os.environ.get("IMAGES_DIR", "images/")

if __name__ == "__main__":
    print(ROOT)
    print(CHECKPOINTS)
    print(DATA)
    print(RESULTS)
