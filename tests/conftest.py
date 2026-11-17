from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import os

os.environ["PYTHONPATH"] = (
    str(Path(__file__).parents[1] / "src")
    + os.pathsep
    + os.environ.get("PYTHONPATH", "")
)
