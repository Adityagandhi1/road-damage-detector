"""Test path setup.

Both roots are added explicitly. `app/` matters because Streamlit page scripts import
`shared` as a top-level module; without it here, tests importing `shared` would only
pass when an AppTest run happened to have inserted the path first.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

for path in (ROOT, ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
