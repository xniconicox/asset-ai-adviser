from __future__ import annotations

import io
import zipfile
from datetime import date
from pathlib import Path

import pytest

from asset_poc.tdnet import (
    _safe_extract_xbrl,
    classify_tdnet_title,
    item_as_dict,
    normalize_tdnet_code,
    parse_tdnet_list,
)

LIST_HTML = b"""
<html><body>
<div onClick="pagerLink('I_list_001_20260812.html')">1</div>
<div onClick="pagerLink('I_list_002_20260812.html')">2</div>
<table id="main-list-table">
<tr>
  <td class="oddnew-L kjTime">16:00</td>
  <td class="oddnew-M kjCode">332A0</td>
  <td class="oddnew-M kjName">G-MEEC</td>
  <td class="oddnew-M kjTitle"><a href="140120260812518429.pdf">Earnings</a></td>
  <td class="oddnew-M kjXbrl"><a href="081220260812518429.zip">XBRL</a></td>
  <td class="oddnew-M kjPlace">TSE</td>
  <td class="oddnew-R kjHistroy"></td>
</tr>
<tr>
  <td class="evennew-L kjTime">15:30</td>
  <td class="evennew-M kjCode">67580</td>
  <td class="evennew-M kjName">Sony</td>
  <td class="evennew-M kjTitle"><a href="140120260812500001.pdf">Notice</a></td>
  <td class="evennew-M kjXbrl"></td>
  <td class="evennew-M kjPlace">TSE</td>
  <td class="evennew-R kjHistroy">updated</td>
</tr>
</table></body></html>
"""


def _zip_bytes(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()


def test_parse_tdnet_list_normalizes_code_and_finds_pages() -> None:
    items, pages = parse_tdnet_list(LIST_HTML, date(2026, 8, 12))

    assert pages == 2
    assert len(items) == 2
    assert item_as_dict(items[0]) == {
        "disclosure_time": "16:00",
        "tdnet_code": "332A0",
        "company_name": "G-MEEC",
        "title": "Earnings",
        "pdf_href": "140120260812518429.pdf",
        "xbrl_href": "081220260812518429.zip",
        "exchange": "TSE",
        "update_history": "",
        "canonical_code": "332A",
        "document_id": "tdnet:140120260812518429",
    }
    assert items[1].canonical_code == "6758"


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("2027年3月期 第1四半期決算短信〔日本基準〕", "earnings_release"),
        ("2026年12月期 決算説明資料", "earnings_presentation"),
        ("通期業績予想の修正に関するお知らせ", "forecast_revision"),
        ("配当予想の変更について", "dividend_revision"),
        ("決算発表日の変更について", "earnings_schedule"),
        ("自己株式取得について", "other"),
    ],
)
def test_classify_tdnet_title(title: str, expected: str) -> None:
    assert classify_tdnet_title(title) == expected


def test_normalize_tdnet_code_preserves_four_character_new_code() -> None:
    assert normalize_tdnet_code("72030") == "7203"
    assert normalize_tdnet_code("332A0") == "332A"
    assert normalize_tdnet_code("332A") == "332A"


def test_safe_extract_xbrl_selects_qualitative_html(tmp_path: Path) -> None:
    content = _zip_bytes(
        {
            "XBRLData/Summary/report.xml": b"<xbrl />",
            "XBRLData/Attachment/qualitative.htm": "<p>本文</p>".encode(),
        }
    )

    destination, html_path = _safe_extract_xbrl(content, tmp_path / "xbrl")

    assert destination == tmp_path / "xbrl"
    assert html_path == tmp_path / "xbrl/XBRLData/Attachment/qualitative.htm"
    assert html_path.read_text() == "<p>本文</p>"


def test_safe_extract_xbrl_rejects_path_traversal(tmp_path: Path) -> None:
    content = _zip_bytes({"../escape.txt": b"bad"})

    with pytest.raises(ValueError, match="Unsafe XBRL"):
        _safe_extract_xbrl(content, tmp_path / "xbrl")
    assert not (tmp_path / "escape.txt").exists()
