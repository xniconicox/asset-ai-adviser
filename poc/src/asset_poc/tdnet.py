from __future__ import annotations

import hashlib
import io
import re
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from pypdf import PdfReader
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from asset_poc.config import Settings
from asset_poc.database import (
    add_manifest,
    connect,
    finish_acquisition_run,
    initialize,
    insert_frame,
    start_acquisition_run,
)
from asset_poc.raw_store import save_raw

TDNET_BASE_URL = "https://www.release.tdnet.info/inbs/"
COLLECTOR_VERSION = "tdnet_free_v1"
MAX_XBRL_FILES = 1_000
MAX_XBRL_UNCOMPRESSED_BYTES = 100 * 1024 * 1024
MAX_EXTRACTED_TEXT_CHARACTERS = 300_000

_FIELD_CLASSES = {
    "kjTime": "disclosure_time",
    "kjCode": "tdnet_code",
    "kjName": "company_name",
    "kjTitle": "title",
    "kjXbrl": "xbrl",
    "kjPlace": "exchange",
    "kjHistroy": "update_history",
}
_EARNINGS_DOCUMENT_TYPES = {
    "earnings_release",
    "earnings_presentation",
    "forecast_revision",
    "dividend_revision",
}


@dataclass(frozen=True)
class TdnetListItem:
    disclosure_time: str
    tdnet_code: str
    company_name: str
    title: str
    pdf_href: str
    xbrl_href: str | None
    exchange: str
    update_history: str

    @property
    def canonical_code(self) -> str:
        return normalize_tdnet_code(self.tdnet_code)

    @property
    def document_id(self) -> str:
        return f"tdnet:{Path(self.pdf_href).stem}"


class _TdnetListParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.items: list[TdnetListItem] = []
        self._row: dict[str, str] | None = None
        self._field: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        if tag == "tr":
            self._row = {}
            return
        if self._row is None:
            return
        if tag == "td":
            classes = set((attributes.get("class") or "").split())
            self._field = next(
                (field for class_name, field in _FIELD_CLASSES.items() if class_name in classes),
                None,
            )
            self._text = []
        elif tag == "a" and self._field in {"title", "xbrl"}:
            href = attributes.get("href")
            if href:
                self._row["pdf_href" if self._field == "title" else "xbrl_href"] = href

    def handle_data(self, data: str) -> None:
        if self._field:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self._field and self._row is not None:
            self._row[self._field] = _normalize_text("".join(self._text))
            self._field = None
            self._text = []
            return
        if tag != "tr" or self._row is None:
            return
        if self._row.get("tdnet_code") and self._row.get("pdf_href"):
            self.items.append(
                TdnetListItem(
                    disclosure_time=self._row.get("disclosure_time", ""),
                    tdnet_code=self._row["tdnet_code"],
                    company_name=self._row.get("company_name", ""),
                    title=self._row.get("title", ""),
                    pdf_href=self._row["pdf_href"],
                    xbrl_href=self._row.get("xbrl_href") or None,
                    exchange=self._row.get("exchange", ""),
                    update_history=self._row.get("update_history", ""),
                )
            )
        self._row = None


class _VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._ignored_depth += 1
        elif tag in {"p", "div", "br", "tr", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in {"p", "div", "tr", "li"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\u3000", " ")).strip()


def normalize_tdnet_code(value: str) -> str:
    code = value.strip().upper().removesuffix(".T")
    return code[:-1] if len(code) == 5 and code.endswith("0") else code


def classify_tdnet_title(title: str) -> str:
    normalized = _normalize_text(title)
    if "決算短信" in normalized or "決算〔" in normalized:
        return "earnings_release"
    if re.search(r"決算(説明|補足|概要|ハイライト)", normalized):
        return "earnings_presentation"
    if "業績予想" in normalized and re.search(r"修正|差異|変更", normalized):
        return "forecast_revision"
    if "配当予想" in normalized and re.search(r"修正|変更", normalized):
        return "dividend_revision"
    if re.search(r"決算発表(日|予定|時刻)", normalized):
        return "earnings_schedule"
    return "other"


def parse_tdnet_list(content: bytes, target_date: date) -> tuple[list[TdnetListItem], int]:
    text = content.decode("utf-8", errors="replace")
    parser = _TdnetListParser()
    parser.feed(text)
    date_token = target_date.strftime("%Y%m%d")
    page_numbers = [
        int(value) for value in re.findall(rf"I_list_(\d{{3}})_{date_token}\.html", text)
    ]
    return parser.items, max(page_numbers, default=1)


def _build_session() -> requests.Session:
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.headers.update(
        {"User-Agent": "asset-ai-adviser-poc/0.1 (research; low-frequency TDnet collector)"}
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _download(session: requests.Session, url: str, *, allow_not_found: bool = False) -> bytes:
    response = session.get(url, timeout=(10, 45))
    if allow_not_found and response.status_code == 404:
        return b""
    response.raise_for_status()
    return response.content


def _save_raw_once(
    connection,
    settings: Settings,
    group: str,
    suffix: str,
    content: bytes,
    source_url: str,
    content_type: str,
    available_at: object | None = None,
) -> tuple[Path, str]:
    digest = hashlib.sha256(content).hexdigest()
    existing = connection.execute(
        "SELECT path FROM raw_manifest WHERE content_hash = ?", [digest]
    ).fetchone()
    if existing and Path(existing[0]).exists():
        return Path(existing[0]), digest
    path, digest = save_raw(settings.raw_dir, group, suffix, content)
    add_manifest(
        connection,
        "tdnet_free_web",
        path,
        digest,
        1,
        source_url=source_url,
        source_tier="A",
        content_type=content_type,
        available_at=available_at,
        collector_version=COLLECTOR_VERSION,
    )
    return path, digest


def _extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    parts = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n\n".join(value for value in parts if value)[:MAX_EXTRACTED_TEXT_CHARACTERS]


def _safe_extract_xbrl(content: bytes, destination: Path) -> tuple[Path, Path | None]:
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        members = archive.infolist()
        if len(members) > MAX_XBRL_FILES:
            raise ValueError(f"XBRL archive has too many files: {len(members)}")
        total_size = sum(member.file_size for member in members)
        if total_size > MAX_XBRL_UNCOMPRESSED_BYTES:
            raise ValueError(f"XBRL archive is too large after extraction: {total_size}")
        for member in members:
            parts = PurePosixPath(member.filename).parts
            file_type = (member.external_attr >> 16) & 0o170000
            if (
                not parts
                or member.filename.startswith(("/", "\\"))
                or ".." in parts
                or file_type == 0o120000
            ):
                raise ValueError(f"Unsafe XBRL archive member: {member.filename}")
        for member in members:
            if member.is_dir():
                continue
            target = destination.joinpath(*PurePosixPath(member.filename).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))

    html_files = sorted(destination.rglob("*.htm")) + sorted(destination.rglob("*.html"))
    qualitative = [path for path in html_files if path.name.lower() == "qualitative.htm"]
    preferred = qualitative or [path for path in html_files if "ixbrl" in path.name.lower()]
    return destination, (preferred or html_files or [None])[0]


def _extract_html_text(path: Path | None) -> str:
    if path is None or not path.exists():
        return ""
    parser = _VisibleTextParser()
    parser.feed(path.read_bytes().decode("utf-8", errors="replace"))
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()[:MAX_EXTRACTED_TEXT_CHARACTERS]


def _available_at(target_date: date, disclosure_time: str) -> pd.Timestamp:
    hour, minute = (int(value) for value in (disclosure_time or "00:00").split(":"))
    disclosed_at = datetime(
        target_date.year,
        target_date.month,
        target_date.day,
        hour,
        minute,
        tzinfo=ZoneInfo("Asia/Tokyo"),
    )
    return pd.Timestamp(disclosed_at).tz_convert("UTC")


def _upsert_metadata(connection, item: TdnetListItem, target_date: date, list_url: str) -> None:
    document_type = classify_tdnet_title(item.title)
    available_at = _available_at(target_date, item.disclosure_time)
    connection.execute(
        """
        INSERT INTO tdnet_documents (
            document_id, canonical_code, tdnet_code, disclosure_date,
            disclosure_time, available_at, company_name, title, exchange,
            document_type, update_history, list_url, pdf_url, xbrl_url,
            status, retrieved_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'listed', ?)
        ON CONFLICT (document_id) DO UPDATE SET
            canonical_code = excluded.canonical_code,
            tdnet_code = excluded.tdnet_code,
            disclosure_date = excluded.disclosure_date,
            disclosure_time = excluded.disclosure_time,
            available_at = excluded.available_at,
            company_name = excluded.company_name,
            title = excluded.title,
            exchange = excluded.exchange,
            document_type = excluded.document_type,
            update_history = excluded.update_history,
            list_url = excluded.list_url,
            pdf_url = excluded.pdf_url,
            xbrl_url = excluded.xbrl_url,
            status = CASE WHEN tdnet_documents.pdf_path IS NULL
                          THEN excluded.status ELSE tdnet_documents.status END,
            retrieved_at = excluded.retrieved_at
        """,
        [
            item.document_id,
            item.canonical_code,
            item.tdnet_code,
            target_date,
            item.disclosure_time,
            available_at,
            item.company_name,
            item.title,
            item.exchange,
            document_type,
            item.update_history,
            list_url,
            urljoin(TDNET_BASE_URL, item.pdf_href),
            None if not item.xbrl_href else urljoin(TDNET_BASE_URL, item.xbrl_href),
            pd.Timestamp.now(tz="UTC"),
        ],
    )


def _enqueue_retry(connection, document_id: str, error: Exception) -> None:
    connection.execute(
        """
        DELETE FROM retry_queue
        WHERE source = 'tdnet' AND item_key = ? AND status = 'pending'
        """,
        [document_id],
    )
    connection.execute(
        """
        INSERT INTO retry_queue VALUES (
            ?, 'tdnet', ?, 'collect_document', 1,
            current_timestamp + INTERVAL 15 MINUTE, 'pending', ?,
            current_timestamp, current_timestamp
        )
        """,
        [str(uuid4()), document_id, str(error)[:4000]],
    )


def _store_document_assets(
    connection,
    settings: Settings,
    session: requests.Session,
    item: TdnetListItem,
    target_date: date,
) -> tuple[int, bool]:
    available_at = _available_at(target_date, item.disclosure_time)
    pdf_url = urljoin(TDNET_BASE_URL, item.pdf_href)
    pdf_content = _download(session, pdf_url)
    if not pdf_content.startswith(b"%PDF"):
        raise ValueError(f"TDnet PDF signature is invalid: {pdf_url}")
    pdf_path, pdf_hash = _save_raw_once(
        connection,
        settings,
        "tdnet_pdf",
        "pdf",
        pdf_content,
        pdf_url,
        "application/pdf",
        available_at,
    )

    xbrl_url: str | None = None
    xbrl_path: Path | None = None
    extract_path: Path | None = None
    html_path: Path | None = None
    xbrl_hash: str | None = None
    if item.xbrl_href:
        xbrl_url = urljoin(TDNET_BASE_URL, item.xbrl_href)
        xbrl_content = _download(session, xbrl_url)
        if not zipfile.is_zipfile(io.BytesIO(xbrl_content)):
            raise ValueError(f"TDnet XBRL archive is invalid: {xbrl_url}")
        xbrl_path, xbrl_hash = _save_raw_once(
            connection,
            settings,
            "tdnet_xbrl",
            "zip",
            xbrl_content,
            xbrl_url,
            "application/zip; contains=xbrl,html",
            available_at,
        )
        extract_path = (
            settings.raw_dir
            / "tdnet_xbrl_extracted"
            / f"{Path(item.pdf_href).stem}_{xbrl_hash[:12]}"
        )
        extract_path, html_path = _safe_extract_xbrl(xbrl_content, extract_path)

    text = _extract_pdf_text(pdf_content)
    if not text:
        text = _extract_html_text(html_path)
    document_type = classify_tdnet_title(item.title)
    retrieved_at = pd.Timestamp.now(tz="UTC")
    if text:
        insert_frame(
            connection,
            "disclosure_texts",
            pd.DataFrame(
                [
                    {
                        "document_id": item.document_id,
                        "canonical_code": item.canonical_code,
                        "disclosure_date": target_date,
                        "disclosure_time": item.disclosure_time,
                        "title": item.title,
                        "document_type": document_type,
                        "source": "tdnet_free_web",
                        "source_url": pdf_url,
                        "raw_path": str(pdf_path.resolve()),
                        "content_hash": pdf_hash,
                        "text_content": text,
                        "retrieved_at": retrieved_at,
                    }
                ]
            ),
        )
    connection.execute(
        """
        UPDATE tdnet_documents SET
            pdf_path = ?, xbrl_path = ?, xbrl_extract_path = ?, html_path = ?,
            pdf_hash = ?, xbrl_hash = ?, text_characters = ?, status = ?,
            retrieved_at = current_timestamp
        WHERE document_id = ?
        """,
        [
            str(pdf_path.resolve()),
            None if xbrl_path is None else str(xbrl_path.resolve()),
            None if extract_path is None else str(extract_path.resolve()),
            None if html_path is None else str(html_path.resolve()),
            pdf_hash,
            xbrl_hash,
            len(text),
            "downloaded" if text else "downloaded_no_text",
            item.document_id,
        ],
    )
    connection.execute(
        """
        UPDATE retry_queue SET status = 'resolved', updated_at = current_timestamp
        WHERE source = 'tdnet' AND item_key = ? AND status = 'pending'
        """,
        [item.document_id],
    )
    return len(text), bool(xbrl_path)


def _validate_date(target_date: date) -> None:
    today = datetime.now(ZoneInfo("Asia/Tokyo")).date()
    if target_date > today:
        raise ValueError("TDnet target date cannot be in the future")
    if target_date < today - timedelta(days=30):
        raise ValueError(
            "TDnet free browsing service exposes only 31 days including the disclosure date"
        )


def collect_tdnet_disclosures(
    settings: Settings,
    target_date: date | str,
    *,
    scope: str = "earnings",
    watchlist_only: bool = True,
    canonical_code: str | None = None,
    limit: int | None = None,
    metadata_only: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Collect a recent TDnet list and selected original PDF/XBRL assets without an API key."""
    target = pd.Timestamp(target_date).date()
    _validate_date(target)
    if scope not in {"earnings", "all"}:
        raise ValueError("scope must be 'earnings' or 'all'")
    settings.ensure_dirs()
    session = _build_session()
    date_token = target.strftime("%Y%m%d")
    first_url = urljoin(TDNET_BASE_URL, f"I_list_001_{date_token}.html")

    with connect(settings.db_path) as connection:
        initialize(connection)
        run_id = start_acquisition_run(
            connection, "tdnet_free_web", f"disclosures:date:{target}:{scope}", 0
        )
        try:
            first_content = _download(session, first_url, allow_not_found=True)
            if not first_content:
                finish_acquisition_run(connection, run_id, "succeeded", 0, 0, "no disclosures")
                return {
                    "target_date": str(target),
                    "market_disclosures": 0,
                    "selected_documents": 0,
                    "downloaded_documents": 0,
                    "run_id": run_id,
                }
            first_items, page_count = parse_tdnet_list(first_content, target)
            _save_raw_once(
                connection,
                settings,
                "tdnet_list",
                "html",
                first_content,
                first_url,
                "text/html; charset=utf-8",
            )
            listed = list(first_items)
            list_urls = {item.document_id: first_url for item in first_items}
            for page in range(2, page_count + 1):
                time.sleep(0.35)
                page_url = urljoin(TDNET_BASE_URL, f"I_list_{page:03d}_{date_token}.html")
                page_content = _download(session, page_url)
                page_items, _ = parse_tdnet_list(page_content, target)
                listed.extend(page_items)
                list_urls.update({item.document_id: page_url for item in page_items})
                _save_raw_once(
                    connection,
                    settings,
                    "tdnet_list",
                    "html",
                    page_content,
                    page_url,
                    "text/html; charset=utf-8",
                )
        except Exception as error:
            finish_acquisition_run(connection, run_id, "failed", 0, 1, str(error))
            raise

        unique = {item.document_id: item for item in listed}
        listed = list(unique.values())
        wanted: set[str] | None = None
        if watchlist_only:
            rows = connection.execute(
                """
                SELECT canonical_code FROM watchlist_membership
                WHERE watchlist_name = 'topix500'
                  AND as_of_date = (SELECT max(as_of_date) FROM watchlist_membership)
                """
            ).fetchall()
            if not rows:
                rows = connection.execute("SELECT canonical_code FROM securities").fetchall()
            wanted = {row[0] for row in rows}
        code_filter = None if canonical_code is None else normalize_tdnet_code(canonical_code)
        selected = [
            item
            for item in listed
            if (wanted is None or item.canonical_code in wanted)
            and (code_filter is None or item.canonical_code == code_filter)
            and (scope == "all" or classify_tdnet_title(item.title) in _EARNINGS_DOCUMENT_TYPES)
        ]
        selected.sort(key=lambda item: (item.disclosure_time, item.document_id), reverse=True)
        if limit is not None:
            selected = selected[:limit]
        connection.execute(
            "UPDATE acquisition_runs SET requested_count = ? WHERE run_id = ?",
            [len(selected), run_id],
        )
        try:
            for item in selected:
                _upsert_metadata(connection, item, target, list_urls[item.document_id])
        except Exception as error:
            finish_acquisition_run(connection, run_id, "failed", 0, 1, str(error))
            raise

        if metadata_only:
            finish_acquisition_run(connection, run_id, "succeeded", len(selected), 0)
            return {
                "target_date": str(target),
                "pages": page_count,
                "market_disclosures": len(listed),
                "selected_documents": len(selected),
                "downloaded_documents": 0,
                "metadata_only": True,
                "run_id": run_id,
            }

        downloaded = 0
        skipped = 0
        xbrl_documents = 0
        text_characters = 0
        errors: list[str] = []
        for index, item in enumerate(selected):
            existing = connection.execute(
                "SELECT pdf_path, status FROM tdnet_documents WHERE document_id = ?",
                [item.document_id],
            ).fetchone()
            if (
                not force
                and existing
                and existing[0]
                and Path(existing[0]).exists()
                and existing[1] in {"downloaded", "downloaded_no_text"}
            ):
                skipped += 1
                continue
            try:
                characters, has_xbrl = _store_document_assets(
                    connection, settings, session, item, target
                )
                downloaded += 1
                text_characters += characters
                xbrl_documents += int(has_xbrl)
            except Exception as error:  # noqa: BLE001 - one disclosure must not stop the day
                errors.append(f"{item.document_id}:{error}")
                connection.execute(
                    "UPDATE tdnet_documents SET status = 'failed' WHERE document_id = ?",
                    [item.document_id],
                )
                _enqueue_retry(connection, item.document_id, error)
            if index < len(selected) - 1:
                time.sleep(0.35)

        success_count = downloaded + skipped
        finish_acquisition_run(
            connection,
            run_id,
            "succeeded" if not errors else "partial",
            success_count,
            len(errors),
            "; ".join(errors)[:4000],
        )
    return {
        "target_date": str(target),
        "pages": page_count,
        "market_disclosures": len(listed),
        "selected_documents": len(selected),
        "downloaded_documents": downloaded,
        "skipped_documents": skipped,
        "xbrl_documents": xbrl_documents,
        "text_characters": text_characters,
        "errors": errors,
        "scope": scope,
        "watchlist_only": watchlist_only,
        "collector_version": COLLECTOR_VERSION,
        "run_id": run_id,
    }


def item_as_dict(item: TdnetListItem) -> dict[str, object]:
    """Expose a stable serializable representation for diagnostics and tests."""
    return {**asdict(item), "canonical_code": item.canonical_code, "document_id": item.document_id}
