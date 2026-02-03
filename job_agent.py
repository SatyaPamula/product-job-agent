import json
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from dateutil import parser as dtparser

import gspread
from google.oauth2.service_account import Credentials


SEARCHAPI_ENDPOINT = "https://www.searchapi.io/api/v1/search"

ROLE_QUERIES = [
    "Senior Product Manager",
    "Product Lead",
    "Group Product Manager",
    "Principal Product Manager",
    "Head of Product",
    "Product Manager",
]

LOCATIONS = [
    ("India", "in"),
    ("Dubai", "ae"),
]

TITLE_REGEX = re.compile(
    r"(senior\s+product\s+manager|principal\s+product\s+manager|group\s+product\s+manager|"
    r"head\s+of\s+product|product\s+lead|product\s+manager)",
    re.IGNORECASE,
)

NEGATIVE_TITLE_REGEX = re.compile(
    r"(intern|internship|junior|associate|scrum|project\s+manager|program\s+manager)",
    re.IGNORECASE,
)

MAX_PAGES_PER_QUERY = 3

STATE_PATH = "jobs/state.json"


@dataclass
class JobRow:
    found_at_utc: str
    posted_at_utc: Optional[str]
    title: str
    company: Optional[str]
    location: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    salary_currency: Optional[str]
    via: Optional[str]
    apply_url: Optional[str]
    source: str
    description_snippet: str


HEADERS = list(JobRow.__dataclass_fields__.keys())


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime]) -> Optional[str]:
    if not dt:
        return None
    return dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def ensure_state_dir() -> None:
    os.makedirs("jobs", exist_ok=True)


def load_state() -> Dict[str, Any]:
    if not os.path.exists(STATE_PATH):
        return {"seen_keys": []}
    with open(STATE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def parse_posted_time(job: Dict[str, Any], now: datetime) -> Optional[datetime]:
    candidates: List[str] = []

    if isinstance(job.get("posted_at"), str):
        candidates.append(job["posted_at"])
    detected = job.get("detected_extensions") or {}
    if isinstance(detected.get("posted_at"), str):
        candidates.append(detected["posted_at"])
    ext = job.get("extensions")
    if isinstance(ext, list):
        candidates.extend([str(x) for x in ext if x])

    if isinstance(job.get("description"), str):
        candidates.append(job["description"][:500])

    text = " | ".join(candidates)

    m = re.search(r"(\d+)\s*(minute|minutes|min|hour|hours|hr|hrs|day|days)\s*ago", text, re.I)
    if m:
        n = int(m.group(1))
        unit = m.group(2).lower()
        if unit.startswith("day"):
            return now - timedelta(days=n)
        if unit.startswith("hour") or unit.startswith("hr"):
            return now - timedelta(hours=n)
        if unit.startswith("min"):
            return now - timedelta(minutes=n)

    try:
        abs_dt = dtparser.parse(text, fuzzy=True)
        if abs_dt.tzinfo is None:
            abs_dt = abs_dt.replace(tzinfo=timezone.utc)
        if abs_dt.year < 2005:
            return None
        return abs_dt.astimezone(timezone.utc)
    except Exception:
        return None


def extract_salary(text: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    if not text:
        return None, None, None

    t = text.replace("–", "-").replace("—", "-")
    m = re.search(
        r"(₹|INR|AED|د\.إ|\$|USD|EUR|€|£|GBP)\s*([\d,\.]+)\s*(?:-|to)\s*"
        r"(₹|INR|AED|د\.إ|\$|USD|EUR|€|£|GBP)?\s*([\d,\.]+)",
        t,
        re.I,
    )
    if not m:
        return None, None, None

    c1 = m.group(1).upper()
    c2 = (m.group(3) or m.group(1)).upper()

    currency_map = {
        "₹": "INR",
        "INR": "INR",
        "AED": "AED",
        "د.إ": "AED",
        "$": "USD",
        "USD": "USD",
        "€": "EUR",
        "EUR": "EUR",
        "£": "GBP",
        "GBP": "GBP",
    }
    cur = currency_map.get(c1, currency_map.get(c2))

    def to_num(x: str) -> Optional[float]:
        try:
            return float(x.replace(",", "").strip())
        except Exception:
            return None

    smin = to_num(m.group(2))
    smax = to_num(m.group(4))
    if smin is not None and smax is not None and smax < smin:
        smin, smax = smax, smin
    return smin, smax, cur


def search_google_jobs(api_key: str, query: str, location: str, gl: str) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    next_token: Optional[str] = None

    for _page in range(MAX_PAGES_PER_QUERY):
        params = {
            "engine": "google_jobs",
            "q": query,
            "location": location,
            "hl": "en",
            "gl": gl,
            "api_key": api_key,
        }
        if next_token:
            params["next_page_token"] = next_token

        r = requests.get(SEARCHAPI_ENDPOINT, params=params, timeout=45)
        r.raise_for_status()
        data = r.json()

        batch = data.get("jobs") or []
        if isinstance(batch, list):
            jobs.extend(batch)

        next_token = data.get("next_page_token")
        if not next_token:
            break

    return jobs


def normalize_jobs(raw_jobs: List[Dict[str, Any]], now: datetime) -> List[JobRow]:
    cutoff = now - timedelta(hours=24)
    out: List[JobRow] = []

    for job in raw_jobs:
        title = (job.get("title") or "").strip()
        if not title:
            continue
        if not TITLE_REGEX.search(title):
            continue
        if NEGATIVE_TITLE_REGEX.search(title):
            continue

        posted_dt = parse_posted_time(job, now)
        if posted_dt and posted_dt < cutoff:
            continue

        company = job.get("company_name")
        location = job.get("location")
        via = job.get("via")

        apply_url = None
        if isinstance(job.get("apply_options"), list) and job["apply_options"]:
            apply_url = job["apply_options"][0].get("link")
        apply_url = apply_url or job.get("link") or job.get("share_link")

        desc = (job.get("description") or "")
        desc_snip = re.sub(r"\s+", " ", desc).strip()[:300]

        smin, smax, cur = extract_salary(desc)

        out.append(
            JobRow(
                found_at_utc=iso(now) or "",
                posted_at_utc=iso(posted_dt),
                title=title,
                company=company,
                location=location,
                salary_min=smin,
                salary_max=smax,
                salary_currency=cur,
                via=via,
                apply_url=apply_url,
                source="SearchApi Google Jobs",
                description_snippet=desc_snip,
            )
        )

    return out


def make_key(r: JobRow) -> str:
    if r.apply_url:
        return r.apply_url.strip()
    return f"{r.title}|{r.company}|{r.location}|{r.posted_at_utc}"


def dedupe(rows: List[JobRow], state: Dict[str, Any]) -> List[JobRow]:
    seen = set(state.get("seen_keys") or [])
    fresh: List[JobRow] = []

    for r in rows:
        k = make_key(r)
        if k in seen:
            continue
        fresh.append(r)
        seen.add(k)

    state["seen_keys"] = list(seen)[-12000:]
    return fresh


def gsheet_client_from_secret() -> gspread.Client:
    raw = os.environ.get("GSHEETS_SERVICE_ACCOUNT_JSON")
    if not raw:
        raise SystemExit("Missing GSHEETS_SERVICE_ACCOUNT_JSON secret")

    sa = json.loads(raw)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(sa, scopes=scopes)
    return gspread.authorize(creds)


def ensure_headers(ws) -> None:
    # If sheet is empty, write headers in row 1
    try:
        first_row = ws.row_values(1)
    except Exception:
        first_row = []

    if first_row != HEADERS:
        ws.clear()
        ws.append_row(HEADERS)


def append_rows(ws, rows: List[JobRow]) -> None:
    values = []
    for r in rows:
        d = asdict(r)
        values.append([d.get(h) for h in HEADERS])

    if values:
        # batch append for speed
        ws.append_rows(values, value_input_option="RAW")


def main() -> None:
    api_key = os.environ.get("SEARCHAPI_KEY")
    spreadsheet_id = os.environ.get("SPREADSHEET_ID")
    sheet_name = os.environ.get("SHEET_NAME", "Jobs")

    if not api_key:
        raise SystemExit("Missing SEARCHAPI_KEY")
    if not spreadsheet_id:
        raise SystemExit("Missing SPREADSHEET_ID")

    ensure_state_dir()

    now = utc_now()

    raw: List[Dict[str, Any]] = []
    for q in ROLE_QUERIES:
        for (loc, gl) in LOCATIONS:
            raw.extend(search_google_jobs(api_key, q, loc, gl))

    normalized = normalize_jobs(raw, now)

    state = load_state()
    fresh = dedupe(normalized, state)
    fresh.sort(key=lambda r: r.posted_at_utc or "", reverse=True)

    # Google Sheets write
    gc = gsheet_client_from_secret()
    sh = gc.open_by_key(spreadsheet_id)
    ws = sh.worksheet(sheet_name)

    ensure_headers(ws)
    append_rows(ws, fresh)

    save_state(state)

    print(f"Fetched raw={len(raw)} | normalized(last24h)={len(normalized)} | appended_new={len(fresh)}")


if __name__ == "__main__":
    main()
