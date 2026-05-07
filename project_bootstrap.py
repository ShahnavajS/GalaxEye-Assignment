from __future__ import annotations

import site
from pathlib import Path


def ensure_local_packages() -> Path:
    repo_root = Path(__file__).resolve().parent
    local_site = repo_root / ".python_packages"
    if local_site.exists():
        site.addsitedir(str(local_site))
    return repo_root
