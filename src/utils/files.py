"""Content-addressed storage for uploaded files.

Streamlit reruns a page's whole script on every widget interaction, and the uploaded
file object survives those reruns. Saving under a fresh UUID each time therefore writes
the entire file again on every slider drag - which grows data/uploads/ without bound and
stalls the UI for anything video-sized.

Naming a file after the hash of its contents makes saving idempotent: the same upload
always resolves to the same path, and the write is skipped when it is already there.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

DEFAULT_SUFFIX = ".bin"
MAX_SUFFIX_LEN = 10


def _safe_suffix(filename: str) -> str:
    """Lowercased extension, or a default.

    Taken from the final path component only, so a name like ``a.jpg/../../b`` cannot
    smuggle separators into the stored filename.
    """
    suffix = Path(str(filename).replace("\\", "/").split("/")[-1]).suffix.lower()

    if not suffix or len(suffix) > MAX_SUFFIX_LEN or not suffix[1:].isalnum():
        return DEFAULT_SUFFIX
    return suffix


def persist_upload(data: bytes, filename: str, directory: str | Path) -> Path:
    """Write ``data`` into ``directory`` under a content-derived name.

    Returns the path. Writes only when the file is not already present, so repeated
    calls with identical bytes are free. The caller's ``filename`` contributes nothing
    but its extension, which also means a hostile name cannot influence the location.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256(data).hexdigest()[:16]
    path = directory / f"{digest}{_safe_suffix(filename)}"

    if not path.exists():
        # Written via a temporary sibling then moved, so a crash mid-write cannot
        # leave a truncated file sitting at the name its hash promises.
        temp = path.with_suffix(path.suffix + ".part")
        temp.write_bytes(data)
        temp.replace(path)

    return path
