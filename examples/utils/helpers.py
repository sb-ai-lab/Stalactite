import os
from pathlib import Path


def path_from_root(path: str) -> str:
    return os.path.normpath(os.path.join(Path(__file__).parent.parent.parent, path))

