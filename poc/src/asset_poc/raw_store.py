from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path


def save_raw(raw_dir: Path, source: str, suffix: str, content: bytes) -> tuple[Path, str]:
    digest = hashlib.sha256(content).hexdigest()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    source_dir = raw_dir / source
    source_dir.mkdir(parents=True, exist_ok=True)
    path = source_dir / f"{stamp}_{digest[:12]}.{suffix.lstrip('.')}"
    path.write_bytes(content)
    return path, digest
