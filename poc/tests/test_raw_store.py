from pathlib import Path

from asset_poc.raw_store import save_raw


def test_raw_store_keeps_content_and_hash(tmp_path: Path) -> None:
    path, digest = save_raw(tmp_path, "example", "json", b'{"ok": true}')
    assert path.read_bytes() == b'{"ok": true}'
    assert digest == "6bc0da1f42f96fc37b8bd7ed20ba57606d2a0da5cda2b135c7854fbdc985b8a3"
