"""Package-wide setup applied before any heavy dependency loads.

Ultralytics ships an "AutoUpdate" behaviour that shells out to pip when it decides an
optional dependency is missing. On a machine where site-packages is not writable this
fails after two slow retries, and the resulting exception surfaces far from its cause -
notably through its patched `PIL.Image.open`, which tries to install `pi-heif` on any
open failure.

Silent self-modifying installs are the wrong default for an app anyway, so they are
turned off here. Declared dependencies live in requirements.txt.
"""

import os

os.environ.setdefault("YOLO_AUTOINSTALL", "false")
