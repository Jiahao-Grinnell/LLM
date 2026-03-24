#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid OCR extraction pipeline for historical slavery/manumission pages.

Design
------
One codebase, three internal stages:
1) extraction   -> high-recall people + event extraction (event-first, old-code style)
2) normalization -> dedupe / clean / merge / order
3) export       -> two CSV outputs matching the newer pipeline

Outputs
-------
1) Detailed info.csv
   Columns: Name, Page, Report Type, Crime Type, Whether abuse, Conflict Type, Trial, Amount paid

2) name place.csv
   Columns: Name, Page, Place, Order, Arrival Date, Date Confidence, Time Info

Optional extras
---------------
- per-page canonical JSON objects for debugging
- run_status.csv with per-page status and model-call counts
- deterministic_only mode for local testing without Ollama
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from difflib import SequenceMatcher

import requests
from requests.exceptions import ConnectionError, ReadTimeout


# ----------------------------- OLLAMA CONFIG -----------------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
REQUEST_TIMEOUT = (10, 600)
MAX_CALL_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1200"))
DEFAULT_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX")
DEFAULT_NUM_CTX = int(DEFAULT_NUM_CTX) if DEFAULT_NUM_CTX and DEFAULT_NUM_CTX.isdigit() else None

_OLLAMA_SESSION = requests.Session()
_OLLAMA_SESSION.headers.update({"Connection": "keep-alive"})


# ----------------------------- DATA MODELS -----------------------------
ALLOWED_EVENT_TYPES = {"presence", "movement", "stay", "manumission", "death_report"}
ALLOWED_DATE_CONF = {"explicit", "derived_from_doc", "unknown"}


@dataclass
class Event:
    type: str
    place: Optional[str] = None
    from_place: Optional[str] = None
    to_place: Optional[str] = None
    date: Optional[str] = None
    date_confidence: str = "unknown"
    evidence: str = ""
    source: str = ""
    time_text: Optional[str] = None

    def canonical_key(self) -> Tuple[Any, ...]:
        return (
            self.type,
            normalize_place(self.place or "") if self.place else "",
            normalize_place(self.from_place or "") if self.from_place else "",
            normalize_place(self.to_place or "") if self.to_place else "",
            normalize_ws(self.date or ""),
            normalize_ws(self.evidence),
        )


@dataclass
class PersonCase:
    name: str
    enslaved_status: str = "strong_inferred"
    enslaved_evidence: str = ""
    events: List[Event] = field(default_factory=list)

    def canonical_name(self) -> str:
        return normalize_name(self.name)


@dataclass
class CaseExtraction:
    doc_id: str
    page: int
    filename: str
    document_date: Optional[str] = None
    people: List[PersonCase] = field(default_factory=list)
    report_type: str = ""
    page_type: str = "unknown"
    status: str = "ok"


# ----------------------------- NORMALIZATION MAPS -----------------------------
PLACE_MAP = {
    "shargah": "Sharjah",
    "sharjah": "Sharjah",
    "sharqah": "Sharjah",
    "shargal": "Sharjah",
    "debai": "Dubai",
    "dibai": "Dubai",
    "dubai": "Dubai",
    "bushire": "Bushehr",
    "bushehr": "Bushehr",
    "bassidu": "Bassidu",
    "bashidu": "Bassidu",
    "basidu": "Bassidu",
    "henjam": "Henjam",
    "qishm": "Qishm",
    "qighm": "Qishm",
    "qiighm": "Qishm",
    "khnjam": "Henjam",
    "majis": "Majis",
    "mekran": "Mekran",
    "mekran coast": "Mekran Coast",
    "mombassa": "Mombasa",
    "mufenesini": "Mufenesini",
    "mkokotoni": "Mkokotoni",
    "zanzibar": "Zanzibar",
    "khassab": "Khassab",
    "muladdha": "Muladdha",
    "wudam": "Wudam",
    "gale": "Gale",
    "ras al khaima": "Ras al Khaimah",
    "ras-al-khaima": "Ras al Khaimah",
    "ras ul khaimah": "Ras al Khaimah",
    "suwaddi isle": "Suwaddi Isle",
    "sur": "Sur",
    "muscat": "Muscat",
    "bahrain": "Bahrain",
    "east africa": "East Africa",
    "arabia": "Arabia",
    "killifi": "Kilifi",
    "elphinstone inlet": "Elphinstone Inlet",
    "lash": "Lash",
    "nogar": "Nogar",
    "dibah": "Dibah",
}

NON_GEO_PLACEHOLDERS = {
    "unknown", "unclear", "n/a", "na", "nil", "none", "agency", "the agency",
    "residency agent", "political agency", "residency", "residency agency", "on board",
    "onboard", "at sea", "shore", "beach", "boat", "dhow", "steamer", "ship",
    "pearl banks", "lowerer", "pearl diver", "town unknown", "district unknown",
}

ROLE_TITLE_PAT = re.compile(
    r"\b(resident|political resident|political agent|residency agent|commanding officer|senior naval officer|wali|chief|shaikh|sheikh|captain|commander|mr\.?|mrs\.?|miss|officer)\b",
    re.I,
)

SHIP_PAT = re.compile(r"\b(h\.?\s*m\.?\s*s\.?|s\.?\s*s\.?|steamer|dhow|boom|sambuk|jallibaut|bagallah|bagarrah)\b", re.I)
BAD_PLACE_TOKEN_PAT = re.compile(
    r"\b(and|or|that|which|who|whom|when|because|therefore|told|replied|served|deserted|ill treatment|profession|owner|present owner|captor|statement|requests|suggest|approved)\b",
    re.I,
)
FULL_NAME_ONLY_PAT = re.compile(r"^[A-Za-z][A-Za-z'’\-]+(?:\s+(?:bin|bint|al|el|ibn)\s+[A-Za-z][A-Za-z'’\-]+|\s+[A-Za-z][A-Za-z'’\-]+){0,6}$")
NAME_WITH_LINEAGE = r"([A-Za-z][A-Za-z'’\-]+(?:\s+(?:bin|bint|ibn|al|el)\s+[A-Za-z][A-Za-z'’\-]+|\s+[A-Za-z][A-Za-z'’\-]+){0,6})"
INDEX_LINE_PAT = re.compile(r"^\s*(?:\(?\d{1,4}\)?[\).]|[A-Z]\.|[ivxlcdm]{1,6}[\).])\s+", re.I)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}
ISO_DATE_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ----------------------------- PROMPTS -----------------------------
EVENT_EXTRACT_PROMPT = """You are an information extraction engine working on historical OCR documents about slavery/manumission in the Persian Gulf region.

GOAL
Extract ONLY enslaved persons (slaves) mentioned in the OCR text and the explicit, evidenced place-linked events for each person.
Your output will be STRICTLY VALIDATED. Do NOT guess, do NOT invent, and do NOT force travel edges.

COMPLETENESS REQUIREMENT
You MUST scan the ENTIRE OCR text and extract ALL place-linked events for each enslaved person.
Common missed items you MUST include when stated:
- "inhabitant of X" / "native of X" / "born at X" / "originally lived at X"
- "lived in X village / in Y district"
- chained moves: "went to X and thence to Y and from there arrived at Z"
- "we/us reached X", "arrived at X", "sent to X", "took me/us to X", "landed at X"
- telegram/list pages where multiple refugee slaves share the same explicit destination or manumission-arrival place

CRITICAL FILTERING
- Include a person ONLY if the text explicitly states OR strongly implies they are enslaved (sold, bought, captured/kidnapped, described as slave, manumitted, requesting manumission, refugee slave).
- Exclude masters/owners, buyers/sellers, officials, rulers, witnesses, and generic roles.

EVENT TYPES
- presence: person explicitly at a place (native of, born at, found at, captured in, sold at, arrived at)
- movement: explicit movement with BOTH origin and destination stated
- stay: explicit residence/stay at a place
- manumission: manumission/certificate/request event, only if the place is explicit
- death_report: death reported at a place, only if the place is explicit

IMPORTANT PLACE RULES
- For any non-movement event, place MUST be a non-empty real place string.
- For movement, NEVER leave from_place or to_place empty.
- Do NOT output vague placeholders like "at sea" or whole sentence fragments as places.
- A ship/dhow name can count as a place only when it is explicitly where the person is/was transported.

DATES
- If a date is explicitly attached to an event, output ISO YYYY-MM-DD if you can do so confidently; otherwise keep the original string.
- If no explicit date is attached, set date=null and date_confidence="unknown".
- NEVER invent dates.

EVIDENCE
Every event must include an evidence quote <=25 words.
Each person must include enslaved_evidence <=25 words.

OUTPUT JSON ONLY:
{{
  "doc_id": "{doc_id}",
  "document_date": null,
  "people": [
    {{
      "name": string,
      "enslaved_status": "explicit"|"strong_inferred",
      "enslaved_evidence": string,
      "events": [
        {{
          "type": "presence"|"movement"|"stay"|"manumission"|"death_report",
          "place": string|null,
          "from_place": string|null,
          "to_place": string|null,
          "date": string|null,
          "date_confidence": "explicit"|"derived_from_doc"|"unknown",
          "evidence": string
        }}
      ]
    }}
  ]
}}

OCR TEXT:
<<<{ocr}>>>
"""

JSON_REPAIR_PROMPT = """The following output was supposed to be valid JSON matching this schema:
{schema}

Repair it into valid JSON only. No markdown. No explanation.

BAD OUTPUT:
<<<{bad}>>>
"""

REPORT_TYPE_PROMPT = """Read the OCR text and classify the page into ONE short report type string.
Possible examples: refugee slave declaration, repatriation telegram, administrative forwarding memo, manumission case summary, approval telegram, cover/index page.
Return JSON only: {{"report_type": string}}

OCR TEXT:
<<<{ocr}>>>
"""


# ----------------------------- BASIC HELPERS -----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def sentence_split(text: str) -> List[str]:
    text = text or ""
    chunks = re.split(r"(?<=[\.!?])\s+|\n{2,}", text)
    return [normalize_ws(c) for c in chunks if normalize_ws(c)]


def sentence_around(text: str, start: int, max_words: int = 25) -> str:
    if not text:
        return ""
    left = max(text.rfind(".", 0, start), text.rfind("\n", 0, start))
    left = 0 if left < 0 else left + 1
    candidates = [p for p in [text.find(".", start), text.find("\n", start)] if p != -1]
    right = min(candidates) if candidates else min(len(text), start + 260)
    snippet = normalize_ws(text[left:right])
    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words])
    return snippet


def choose_preferred_name(a: str, b: str) -> str:
    a = normalize_name(a)
    b = normalize_name(b)
    if not a:
        return b
    if not b:
        return a
    score_a = (len(a.split()), len(a), int(" bin " in f" {a.lower()} "))
    score_b = (len(b.split()), len(b), int(" bin " in f" {b.lower()} "))
    return b if score_b > score_a else a


def normalize_name(raw: str) -> str:
    if not raw:
        return ""
    s = raw
    s = re.split(r"\|\s*role\s*:", s, flags=re.I)[0]
    s = re.sub(r"\([^)]*\)", "", s)
    s = re.split(r"\b(?:who|which|that|requests?|requesting|formerly|being|arriving|for\s+manumission|as\s+slave\s+no\.?|slave\s+no\.?)\b", s, maxsplit=1, flags=re.I)[0]
    s = re.sub(r"\b(?:aged\s+about|age\s+about|age\s+unknown|judged\s+about|small\s+son\s+aged|eldest\s+son|his\s+mark|her\s+mark)\b.*$", "", s, flags=re.I)
    s = normalize_ws(s).strip(" ,.;:[]{}\"'")
    s = re.sub(r"\b(ibn|bin)\b", "bin", s, flags=re.I)
    s = re.sub(r"\b(bint)\b", "bint", s, flags=re.I)
    s = re.sub(r"\b(abu)\b", "Abu", s, flags=re.I)
    s = re.sub(r"\b(umm)\b", "Umm", s, flags=re.I)
    s = re.sub(r"\b([A-Za-z]{2,})r?aged\b.*$", r"\1", s, flags=re.I)
    s = normalize_ws(s)
    return s


def is_likely_personal_name(name: str) -> bool:
    if not name:
        return False
    n = normalize_ws(name)
    if len(n) < 2:
        return False
    low = n.lower()
    if low in {"unknown", "unnamed", "slave", "woman", "man", "boy", "girl", "children", "others", "dated the", "of", "no", "telegram code"}:
        return False
    if any(tok in {"dated", "received", "telegram", "from", "to", "your", "suggestion", "approved", "resident"} for tok in low.split()) and len(low.split()) <= 2:
        return False
    if ROLE_TITLE_PAT.search(n) and len(n.split()) <= 6:
        return False
    if re.search(r"\d", n):
        return False
    return True


def normalize_ship_name(text: str) -> str:
    s = normalize_ws(text)
    s = re.sub(r"\bH\s*\.?\s*M\s*\.?\s*S\s*\.?\b", "H.M.S.", s, flags=re.I)
    s = re.sub(r"\bS\s*\.?\s*S\s*\.?\b", "S.S.", s, flags=re.I)
    return s


def is_ship(place: str) -> bool:
    return bool(SHIP_PAT.search(place or ""))


def normalize_place(raw: str) -> str:
    if not raw:
        return ""
    s = normalize_ws(raw).strip(" ,.;:[]{}\"'")
    if not s:
        return ""
    if is_ship(s):
        return normalize_ship_name(s)
    s = re.sub(r"^(?:the\s+)?(?:town|village|district|coast|port|island)\s+of\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:at|in|to|from|near|off|on)\s+", "", s, flags=re.I)
    s = re.split(
        r"\b(?:named|where|who|whom|which|when|that|while|during|after|before|because|for\s+delivery|recorded|signed|dated|having|leaving|left|being\s+sent|intended\s+to|wanted\s+to)\b",
        s,
        maxsplit=1,
        flags=re.I,
    )[0]
    s = normalize_ws(s.strip(" ,.;:"))
    if not s:
        return ""
    low = s.lower().replace("-", " ")
    low = re.sub(r"\bras\s+ul\b", "ras al", low)
    low = re.sub(r"\bul\b", "al", low)
    low = re.sub(r"\bel\b", "al", low)
    low = low.replace("mlphinstone", "elphinstone")
    low = normalize_ws(low)
    mapped = PLACE_MAP.get(low)
    if mapped:
        return mapped
    if len(s) <= 4 and s.isupper():
        return s
    return " ".join(w[:1].upper() + w[1:] if w else w for w in low.split())


def is_valid_place(place: str) -> bool:
    if not place:
        return False
    p = normalize_ws(place)
    if not p:
        return False
    if is_ship(p):
        return True
    low = p.lower()
    if low in NON_GEO_PLACEHOLDERS or low in {"there", "here", "this place", "that place"}:
        return False
    if re.match(r"^(?:the\s+)?house\s+of\b", low):
        return False
    if re.match(r"^(?:ruler|chief|shaikh|sheikh|wali|resident|political\s+agent|residency\s+agent)\s+of\b", low):
        return False
    if re.search(r"\d", p):
        return False
    tokens = [t for t in p.split() if t]
    if len(tokens) > 6:
        return False
    if BAD_PLACE_TOKEN_PAT.search(low) and len(tokens) >= 4:
        return False
    return True


def extract_year_from_document_date(doc_date: Optional[str]) -> Optional[int]:
    if not doc_date:
        return None
    m = re.search(r"\b(17|18|19|20)\d{2}\b", doc_date)
    return int(m.group(0)) if m else None


def parse_day_month(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    s = normalize_ws(text.lower().replace(",", " "))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\b", s)
    if not m:
        return None
    day = int(m.group(1))
    mon_name = m.group(2)
    for mon, num in MONTHS.items():
        if mon_name.startswith(mon[:3]):
            return day, num
    return None


def to_iso_date(date_str: Optional[str], doc_year: Optional[int] = None) -> Tuple[Optional[str], str]:
    if not date_str:
        return None, "unknown"
    s = normalize_ws(date_str)
    if not s:
        return None, "unknown"
    if ISO_DATE_PAT.match(s):
        return s, "explicit"
    m = re.search(r"(?:\bD/?\s*)?(\d{1,2})-(\d{1,2})-(\d{2})\b", s, flags=re.I)
    if m and doc_year:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        century = (doc_year // 100) * 100
        year = century + yy
        if abs(year - doc_year) <= 10 and 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{year:04d}-{mm:02d}-{dd:02d}", "derived_from_doc"
    dm = parse_day_month(s)
    if dm and doc_year:
        dd, mm = dm
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{doc_year:04d}-{mm:02d}-{dd:02d}", "derived_from_doc"
    return s, "unknown"


def extract_document_date(text: str) -> Optional[str]:
    if not text:
        return None
    for pat in [
        r"\bDated(?:\s+and\s+received)?\s+the\s+([^\.\n]{4,60})",
        r"\bDATE\.*\s*([^\n]{4,60})",
        r"\bRecorded\s+on\s+([^\.\n]{4,60})",
        r"\bdated\s+([^\.\n]{4,60})",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_ws(m.group(1))
    return None


def chunk_text_by_lines(text: str, lines_per_chunk: int = 70) -> List[str]:
    lines = text.splitlines()
    chunks: List[str] = []
    buf: List[str] = []
    for line in lines:
        buf.append(line)
        if len(buf) >= lines_per_chunk:
            c = "\n".join(buf).strip()
            if c:
                chunks.append(c)
            buf = []
    if buf:
        c = "\n".join(buf).strip()
        if c:
            chunks.append(c)
    return chunks


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = min([i for i in [text.find("{"), text.find("[")] if i != -1], default=-1)
    if start == -1:
        return None
    stack: List[str] = []
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch in "[{":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                continue
            opener = stack.pop()
            if (opener, ch) not in {("{", "}"), ("[", "]")}:
                return None
            if not stack:
                candidate = text[start:idx + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


# ----------------------------- PAGE CLASSIFICATION -----------------------------
def looks_like_index_page(ocr: str) -> bool:
    if not ocr:
        return False
    lines = [ln.strip() for ln in ocr.splitlines() if ln.strip()]
    if len(lines) < 20:
        return False
    sample = lines[:160]
    enum = sum(1 for ln in sample if INDEX_LINE_PAT.match(ln))
    short = sum(1 for ln in sample if len(ln) <= 100)
    low = "\n".join(sample).lower()
    has_header = "contents" in low or "index" in low or "individual cases" in low
    return (enum >= 10 and short / max(1, len(sample)) >= 0.60) or (has_header and enum >= 6 and short / max(1, len(sample)) >= 0.55)


def looks_like_record_metadata_page(ocr: str) -> bool:
    if not ocr:
        return False
    head = normalize_ws(ocr[:2600]).lower()
    hits = sum(
        1 for pat in [
            "qatar digital library", "about this record", "the online record can be viewed at",
            "holding institution", "extent and format", "copyright for document",
            "open government licence", "not for direct photocopying",
        ] if pat in head
    )
    return hits >= 3


def score_ocr_quality(text: str) -> Tuple[str, Dict[str, Any]]:
    if not text:
        return "garbled", {"strong_anchor": False}
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_alpha_lines = sum(1 for ln in lines[:80] if re.search(r"[A-Za-z]", ln) and len(re.findall(r"[A-Za-z]", ln)) < 12)
    repeated_line_ratio = 1.0 - (len(set(lines[:80])) / max(1, len(lines[:80]))) if lines[:80] else 0.0
    tokens = re.findall(r"\S+", text)
    weird_tokens = sum(1 for t in tokens if re.search(r"[^A-Za-z0-9,.;:'’\-\(\)/]", t))
    weird_ratio = weird_tokens / max(1, len(tokens))
    strong_anchor = bool(re.search(
        r"\b(statement\s+made\s+by|refugee\s+slaves|slave\s*no\.?\s*\d+|named\s+[A-Z]|manumission\s+certificate|requests\s+repatriation)\b",
        text,
        flags=re.I,
    ))
    info = {
        "strong_anchor": strong_anchor,
        "repeated_line_ratio": round(repeated_line_ratio, 3),
        "weird_ratio": round(weird_ratio, 3),
        "short_line_ratio": round(short_alpha_lines / max(1, len(lines[:80])), 3),
    }
    if weird_ratio >= 0.20 and not strong_anchor:
        return "garbled", info
    if repeated_line_ratio >= 0.12 or weird_ratio >= 0.12:
        return "salvageable", info
    return "normal", info


def classify_page(text: str) -> str:
    low = (text or "").lower()
    if looks_like_index_page(text):
        return "index"
    if looks_like_record_metadata_page(text):
        return "record_metadata"
    if "summaries of declarations" in low or re.search(r"\bno\.\s*\d+\.?\s+[A-Z]", text):
        return "declaration_summary"
    if re.search(r"\bstatement\s+made\s+by\b", low):
        return "narrative_statement"
    if re.search(r"\btelegram\b", low):
        return "telegram"
    if re.search(r"\bforwarded\s+to\b|\bfor\s+manumission\b|\bgrant\s+.*manumission\b", low):
        return "administrative_route"
    return "unknown"


# ----------------------------- OLLAMA CLIENT -----------------------------
def call_ollama(prompt: str, stats: Dict[str, int], *, num_predict: int = DEFAULT_NUM_PREDICT,
                num_ctx: Optional[int] = DEFAULT_NUM_CTX) -> str:
    last_err: Optional[Exception] = None
    for attempt in range(1, MAX_CALL_RETRIES + 1):
        try:
            payload = {
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": int(num_predict)},
            }
            if num_ctx:
                payload["options"]["num_ctx"] = int(num_ctx)
            stats["model_calls"] += 1
            resp = _OLLAMA_SESSION.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            return (resp.json().get("response") or "").strip()
        except (ReadTimeout, ConnectionError, requests.HTTPError) as e:
            last_err = e
            if attempt < MAX_CALL_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    raise RuntimeError(f"Ollama call failed after retries: {last_err}")


def call_json_prompt(prompt: str, stats: Dict[str, int], schema_hint: str, *, num_predict: int = DEFAULT_NUM_PREDICT) -> Optional[Any]:
    raw = call_ollama(prompt, stats, num_predict=num_predict)
    parsed = extract_json(raw)
    if parsed is not None:
        stats["extract_calls"] += 1
        return parsed
    repaired = call_ollama(JSON_REPAIR_PROMPT.format(schema=schema_hint, bad=raw), stats, num_predict=800)
    stats["repair_calls"] += 1
    return extract_json(repaired)


# ----------------------------- EXTRACTION: DETERMINISTIC NAMES -----------------------------
def _add_name_candidate(out: Dict[str, Dict[str, str]], raw_name: str, evidence: str) -> None:
    name = normalize_name(raw_name)
    if not is_likely_personal_name(name):
        return
    evidence = normalize_ws(evidence)
    if not evidence:
        return
    key = name.lower()
    prev = out.get(key)
    if not prev or len(evidence) > len(prev.get("evidence", "")) or len(name) > len(prev.get("name", "")):
        out[key] = {"name": name, "evidence": evidence}


def _extract_name_from_piece(piece: str) -> str:
    piece = normalize_ws(piece)
    if not piece:
        return ""
    piece = re.sub(r"^\(?\d+\)?[.):-]?\s*", "", piece)
    piece = re.sub(r"^slave\s*no\.?\s*\d+\s*[:=\-]\s*", "", piece, flags=re.I)
    piece = re.sub(r"\([^)]*\)", "", piece)
    piece = re.sub(r"\b(?:one\s+swahili|three\s+baluchis?|four\s+refugee\s+slaves|young\s+swahili|small\s+son|eldest\s+son)\b", "", piece, flags=re.I)
    piece = re.sub(r"\b(?:aged\s+about|born\s+at|native\s+of|recorded\s+on|served\s+\d+\s+years?)\b.*$", "", piece, flags=re.I)
    piece = piece.strip(" ,.;:")
    if FULL_NAME_ONLY_PAT.match(piece):
        return normalize_name(piece)
    return ""


def deterministic_named_people(text: str) -> List[Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    body = text or ""
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    # line-oriented patterns are much safer on this corpus than global regexes
    for line in lines:
        m = re.search(rf"\bnamed\s+{NAME_WITH_LINEAGE}", line, flags=re.I)
        if m:
            _add_name_candidate(out, m.group(1), line)

        m = re.search(rf"^No\.\s*\d+\.?\s*{NAME_WITH_LINEAGE}", line, flags=re.I)
        if m:
            _add_name_candidate(out, m.group(1), line)

        m = re.search(rf"^Slave\s+no\.\s*\d+\s*[\-:=]\s*{NAME_WITH_LINEAGE}", line, flags=re.I)
        if m:
            _add_name_candidate(out, m.group(1), line)

        # refugee-slave list lines
        m = re.search(r"\brefugee\s+slaves?\s+namely\s+(.+)$", line, flags=re.I)
        if not m:
            m = re.search(r"\bthe\s+following\s+refugee\s+slaves[^:]*:\s*(.+)$", line, flags=re.I)
        if m:
            seg = m.group(1)
            seg = re.sub(r"\bone\s+swahili\b", "", seg, flags=re.I)
            seg = re.sub(r"\btwo\s+swahilis?\b", "", seg, flags=re.I)
            seg = re.sub(r"\bthree\s+baluchis?\b", "", seg, flags=re.I)
            seg = re.sub(r"\bfour\s+refugee\s+slaves\b", "", seg, flags=re.I)
            seg = re.split(r"\b(?:are\s+being|arriving|letter\s+accompanies|please\s+arrange|for\s+manumission\b)\b", seg, maxsplit=1, flags=re.I)[0]
            seg = seg.replace(" and ", ", ")
            for piece in [p.strip() for p in seg.split(",") if p.strip()]:
                nm = _extract_name_from_piece(piece)
                if nm:
                    _add_name_candidate(out, nm, line)

    return list(out.values())


# ----------------------------- EXTRACTION: NAME-SCOPED CONTEXT -----------------------------
def extract_numbered_block(text: str, name: str) -> str:
    target = normalize_name(name).lower()
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        low = normalize_name(line).lower()
        if re.search(r"\bno\.\s*\d+", line, flags=re.I) and target and target in low:
            start_idx = i
            break
    if start_idx is None:
        return ""
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if re.search(r"^\s*No\.\s*\d+", lines[j], flags=re.I):
            end_idx = j
            break
    return normalize_ws("\n".join(lines[start_idx:end_idx]))


def extract_lineage_block(text: str, name: str) -> str:
    target = normalize_name(name).lower()
    lines = text.splitlines()
    start_idx = None
    for i, line in enumerate(lines):
        low = normalize_name(line).lower()
        if re.search(r"slave\s+no\.?\s*\d+", line, flags=re.I) and target and target in low:
            start_idx = i
            break
    if start_idx is None:
        return ""
    end_idx = len(lines)
    for j in range(start_idx + 1, len(lines)):
        if re.search(r"^\s*Slave\s+no\.?\s*\d+", lines[j], flags=re.I):
            end_idx = j
            break
    return normalize_ws("\n".join(lines[start_idx:end_idx]))


def extract_focus_text_for_name(text: str, name: str) -> str:
    text = text or ""
    target = normalize_name(name)
    if not text or not target:
        return text

    for getter in (extract_numbered_block, extract_lineage_block):
        block = getter(text, target)
        if block:
            return block

    name_re = re.escape(target).replace("\\ ", r"\s+")
    sentences = sentence_split(text)
    selected: List[str] = []
    for idx, sent in enumerate(sentences):
        if re.search(rf"\b{name_re}\b", sent, flags=re.I):
            if idx - 1 >= 0:
                selected.append(sentences[idx - 1])
            selected.append(sent)
            if idx + 1 < len(sentences):
                selected.append(sentences[idx + 1])
    if selected:
        dedup: List[str] = []
        seen = set()
        for s in selected:
            if s not in seen:
                dedup.append(s)
                seen.add(s)
        return "\n\n".join(dedup)
    return text


def detect_group_destination_events(text: str, names: Sequence[str], doc_year: Optional[int]) -> Dict[str, List[Event]]:
    shared: Dict[str, List[Event]] = {normalize_name(n): [] for n in names}
    if not text or not names:
        return shared
    for m in re.finditer(r"\b(?:arriving|arrive)\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:about|on|by|,|\.|;|$))", text, flags=re.I):
        place = normalize_place(m.group(1))
        if not is_valid_place(place):
            continue
        ev = Event(
            type="presence",
            place=place,
            date=None,
            date_confidence="unknown",
            evidence=sentence_around(text, m.start()),
            source="deterministic_group",
        )
        for n in names:
            shared[normalize_name(n)].append(ev)
    return shared


# ----------------------------- EXTRACTION: DETERMINISTIC EVENTS -----------------------------
def _event_presence(events: List[Event], place: str, evidence: str, *, source: str, date: Optional[str] = None,
                    date_confidence: str = "unknown", time_text: Optional[str] = None) -> None:
    place = normalize_place(place)
    if not is_valid_place(place):
        return
    events.append(Event(
        type="presence",
        place=place,
        date=date,
        date_confidence=date_confidence,
        evidence=" ".join(normalize_ws(evidence).split()[:25]),
        source=source,
        time_text=time_text,
    ))


def _event_movement(events: List[Event], from_place: str, to_place: str, evidence: str, *, source: str,
                    date: Optional[str] = None, date_confidence: str = "unknown", time_text: Optional[str] = None) -> None:
    fp = normalize_place(from_place)
    tp = normalize_place(to_place)
    if not (is_valid_place(fp) and is_valid_place(tp)):
        return
    events.append(Event(
        type="movement",
        from_place=fp,
        to_place=tp,
        date=date,
        date_confidence=date_confidence,
        evidence=" ".join(normalize_ws(evidence).split()[:25]),
        source=source,
        time_text=time_text,
    ))


def extract_explicit_events_for_name(text: str, name: str, doc_year: Optional[int]) -> List[Event]:
    focus = extract_focus_text_for_name(text, name)
    if not focus:
        return []
    events: List[Event] = []
    sentences = sentence_split(focus)
    for sent in sentences:
        low = sent.lower()

        def add_presence(pattern: str) -> None:
            m = re.search(pattern, sent, flags=re.I)
            if m:
                _event_presence(events, m.group(1), sent, source="deterministic")

        def add_movement(pattern: str) -> None:
            m = re.search(pattern, sent, flags=re.I)
            if m:
                _event_movement(events, m.group(1), m.group(2), sent, source="deterministic")

        add_presence(r"\bnative\s+of\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|Age|with|$))")
        add_presence(r"\bborn\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\(|Age|Shipped|$))")
        if 'father' not in low and 'mother' not in low:
            add_presence(r"\bborn\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\(|Age|Shipped|$))")
        add_presence(r"\boriginally\s+lived\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\sas\s+slaves|$))")
        add_presence(r"\blived\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|$))")
        add_presence(r"\bpresent\s+owner[^\.]{0,120}?\bof\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:\(|,|\.|;|$))")
        add_presence(r"\bfirst\s+owner[^\.]{0,120}?\bof\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:\(|,|\.|;|$))")
        add_presence(r"\bsecond[^\.]{0,120}?\bof\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:\(|,|\.|;|$))")
        add_presence(r"\bResidency\s+Agent\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\s+but|$))")
        add_presence(r"\bshipped\s+from\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:\d|years?|,|\.|;|$))")
        add_presence(r"\blanded\s+and\s+sold\s+(?:near|at)\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|$))")
        add_presence(r"\blanded\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:by|,|\.|;|$))")
        add_presence(r"\bhad\s+fled\s+from\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:on|in|,|\.|;|$))")
        add_presence(r"\bmoved\s+to\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|where|$))")
        add_presence(r"\brequests?\s+repatriation\s+to\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|$))")
        add_presence(r"\barriv(?:ed|ing)\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:about|on|by|,|\.|;|$))")
        add_presence(r"\bto\s+seek\s+manumission\s+from\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|$))")

        add_movement(r"\bwas\s+kidnapped[^\.]{0,80}?\bfrom\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:and|to|by|where|$|[,.]))[^\.]{0,120}?\bto\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:where|by|for|\(|,|\.|;|$))")
        add_movement(r"\bsailed\s+from\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:and|,|\.|;|$))[^\.]{0,80}?\blanded\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:by|,|\.|;|$))")
        add_movement(r"\bfrom\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+and\s+taken|\s+to)[^\.]{0,120}?\bto\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s+(?:where|by|for|\(|,|\.|;|$))")

    return events


def extract_name_evidence(text: str, name: str) -> str:
    focus = extract_focus_text_for_name(text, name)
    for pat in [
        rf"\bslave\b[^\.\n]{{0,80}}\b{name}\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\bmanumission\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\bdeserted\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\brequests?\s+repatriation\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\brequest(?:ing)?\s+to\s+be\s+manumitted\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\bsold\b",
        rf"\b{name}\b[^\.\n]{{0,80}}\bkidnapped\b",
    ]:
        m = re.search(pat, focus, flags=re.I)
        if m:
            return sentence_around(focus, m.start())
    return sentence_around(focus, 0)


# ----------------------------- EXTRACTION: MODEL PASS -----------------------------
def parse_model_people(obj: Any) -> List[PersonCase]:
    people: List[PersonCase] = []
    if not isinstance(obj, dict):
        return people
    for p in obj.get("people") or []:
        if not isinstance(p, dict):
            continue
        name = normalize_name(str(p.get("name") or ""))
        if not is_likely_personal_name(name):
            continue
        person = PersonCase(
            name=name,
            enslaved_status=str(p.get("enslaved_status") or "strong_inferred"),
            enslaved_evidence=normalize_ws(str(p.get("enslaved_evidence") or ""))[:300],
            events=[],
        )
        for ev in p.get("events") or []:
            if not isinstance(ev, dict):
                continue
            e = Event(
                type=str(ev.get("type") or ""),
                place=ev.get("place"),
                from_place=ev.get("from_place"),
                to_place=ev.get("to_place"),
                date=ev.get("date"),
                date_confidence=str(ev.get("date_confidence") or "unknown"),
                evidence=normalize_ws(str(ev.get("evidence") or "")),
                source="model",
            )
            person.events.append(e)
        people.append(person)
    return people


def model_extract_case(ocr: str, doc_id: str, stats: Dict[str, int]) -> List[PersonCase]:
    schema_hint = '{"doc_id":"%s","document_date":null,"people":[{"name":"X","enslaved_status":"explicit","enslaved_evidence":"...","events":[{"type":"presence","place":"Sharjah","from_place":null,"to_place":null,"date":null,"date_confidence":"unknown","evidence":"..."}]}]}' % doc_id
    chunks = chunk_text_by_lines(ocr, lines_per_chunk=70)
    objs: List[Any] = []
    if len(chunks) <= 1:
        obj = call_json_prompt(EVENT_EXTRACT_PROMPT.format(doc_id=doc_id, ocr=ocr), stats, schema_hint, num_predict=1300)
        if obj is not None:
            objs.append(obj)
    else:
        for idx, chunk in enumerate(chunks, 1):
            obj = call_json_prompt(EVENT_EXTRACT_PROMPT.format(doc_id=f"{doc_id}#chunk{idx}", ocr=chunk), stats, schema_hint, num_predict=1000)
            if obj is not None:
                objs.append(obj)
    out_people: List[PersonCase] = []
    for obj in objs:
        out_people.extend(parse_model_people(obj))
    return out_people


def infer_report_type(ocr: str, stats: Dict[str, int], deterministic_only: bool) -> str:
    low = (ocr or "").lower()
    if "summaries of declarations" in low:
        return "refugee slave declaration"
    if "repatriation" in low:
        return "repatriation telegram"
    if "suggestion approved" in low:
        return "approval telegram"
    if "for manumission" in low and "telegram" in low:
        return "administrative forwarding memo"
    if deterministic_only:
        return "historical slavery/manumission page"
    try:
        obj = call_json_prompt(REPORT_TYPE_PROMPT.format(ocr=ocr[:7000]), stats, '{"report_type":"string"}', num_predict=80)
        if isinstance(obj, dict) and normalize_ws(str(obj.get("report_type") or "")):
            return normalize_ws(str(obj.get("report_type")))
    except Exception:
        pass
    return "historical slavery/manumission page"


# ----------------------------- NORMALIZATION / MERGE -----------------------------
def coerce_enslaved_status(status: str, evidence: str) -> str:
    status = status if status in {"explicit", "strong_inferred"} else "strong_inferred"
    if status == "explicit" and not re.search(r"\b(sold|bought|slave|captur|kidnap|manumit|owned|belong|purchase|refugee\s+slave)\b", evidence or "", re.I):
        return "strong_inferred"
    return status


def normalize_event(ev: Event, doc_year: Optional[int]) -> Optional[Event]:
    if ev.type not in ALLOWED_EVENT_TYPES:
        return None
    date, conf = to_iso_date(ev.date, doc_year)
    evidence = normalize_ws(ev.evidence)
    if not evidence:
        return None
    if len(evidence.split()) > 28:
        evidence = " ".join(evidence.split()[:25])
    if ev.type == "movement":
        fp = normalize_place(ev.from_place or "")
        tp = normalize_place(ev.to_place or "")
        if not (is_valid_place(fp) and is_valid_place(tp)):
            return None
        return Event(
            type="movement",
            from_place=fp,
            to_place=tp,
            date=date,
            date_confidence=conf if conf in ALLOWED_DATE_CONF else "unknown",
            evidence=evidence,
            source=ev.source,
            time_text=normalize_ws(ev.time_text or "") or None,
        )
    pl = normalize_place(ev.place or "")
    if not is_valid_place(pl):
        return None
    return Event(
        type=ev.type,
        place=pl,
        date=date,
        date_confidence=conf if conf in ALLOWED_DATE_CONF else "unknown",
        evidence=evidence,
        source=ev.source,
        time_text=normalize_ws(ev.time_text or "") or None,
    )


def names_maybe_same_person(a: str, b: str) -> bool:
    a_n = normalize_name(a).lower()
    b_n = normalize_name(b).lower()
    if not a_n or not b_n:
        return False
    if a_n == b_n or a_n.replace(" ", "") == b_n.replace(" ", ""):
        return True
    a_toks = [t for t in a_n.split() if t not in {"bin", "bint", "al", "el"}]
    b_toks = [t for t in b_n.split() if t not in {"bin", "bint", "al", "el"}]
    if not a_toks or not b_toks:
        return False
    if a_toks == b_toks:
        return True
    if a_toks[0] == b_toks[0] and (a_toks[-1] == b_toks[-1] or len(set(a_toks) & set(b_toks)) >= 2):
        return True
    if SequenceMatcher(None, ''.join(a_toks), ''.join(b_toks)).ratio() >= 0.90:
        return True
    return False


def stable_sort_events_by_date(events: List[Event]) -> List[Event]:
    dated: List[Tuple[str, int, Event]] = []
    positions: List[int] = []
    for idx, ev in enumerate(events):
        if ev.date and ISO_DATE_PAT.match(ev.date):
            dated.append((ev.date, idx, ev))
            positions.append(idx)
    dated_sorted = [ev for _, _, ev in sorted(dated, key=lambda x: (x[0], x[1]))]
    out = list(events)
    it = iter(dated_sorted)
    for pos in positions:
        out[pos] = next(it)
    return out


def merge_people(det_people: List[PersonCase], model_people: List[PersonCase], doc_year: Optional[int]) -> List[PersonCase]:
    merged: List[PersonCase] = []
    for src_people in [det_people, model_people]:
        for p in src_people:
            name = normalize_name(p.name)
            if not is_likely_personal_name(name):
                continue
            match = None
            for existing in merged:
                if names_maybe_same_person(existing.name, name):
                    match = existing
                    break
            if match is None:
                match = PersonCase(name=name, enslaved_status="strong_inferred", enslaved_evidence="", events=[])
                merged.append(match)
            match.name = choose_preferred_name(match.name, name)
            status = coerce_enslaved_status(p.enslaved_status, p.enslaved_evidence)
            if status == "explicit":
                match.enslaved_status = "explicit"
            if len(p.enslaved_evidence or "") > len(match.enslaved_evidence or ""):
                match.enslaved_evidence = p.enslaved_evidence
            for ev in p.events:
                nev = normalize_event(ev, doc_year)
                if nev is not None:
                    match.events.append(nev)

    for p in merged:
        dedup: List[Event] = []
        seen = set()
        for ev in p.events:
            key = ev.canonical_key()
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ev)
        p.events = stable_sort_events_by_date(dedup)
    merged = [p for p in merged if p.enslaved_evidence and (p.events or p.enslaved_evidence)]
    return merged


def events_to_place_timeline(events: List[Event]) -> List[Tuple[str, Optional[str], str, Optional[str]]]:
    rows: List[Tuple[str, Optional[str], str, Optional[str]]] = []
    last_place = None
    for ev in events:
        if ev.type == "movement":
            for pl in [ev.from_place, ev.to_place]:
                if pl and pl != last_place:
                    rows.append((pl, ev.date, ev.date_confidence, ev.time_text))
                    last_place = pl
        else:
            pl = ev.place
            if pl and pl != last_place:
                rows.append((pl, ev.date, ev.date_confidence, ev.time_text))
                last_place = pl
    return rows


# ----------------------------- METADATA FOR DETAILED CSV -----------------------------
def extract_name_scoped_context(ocr: str, name: str) -> str:
    return extract_focus_text_for_name(ocr, name)


def find_amount_paid(text: str) -> str:
    if not text:
        return ""
    for pat in [
        r"\bpaid\s+(?:him|her|them)?\s*(?:the\s+sum\s+of\s+)?([A-Za-z0-9\- ]{2,30})",
        r"\bsum\s+of\s+([A-Za-z0-9\- ]{2,30})",
    ]:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_ws(m.group(1))
    return ""


def deterministic_meta(ocr: str, name: str, page: int, report_type: str) -> Dict[str, Any]:
    scoped = extract_name_scoped_context(ocr, name)
    low = scoped.lower()
    crime_type = ""
    if re.search(r"\b(kidnapped|abducted)\b", low):
        crime_type = "kidnapping"
    elif re.search(r"\btraffick", low):
        crime_type = "trafficking"
    elif re.search(r"\bsold\b", low):
        crime_type = "sale"

    whether_abuse = ""
    if re.search(r"\b(beat|beaten|abused|ill-treated|ill treatment|forced|cruel|violence|assault|maltreat|manacles?|imprison)\b", low):
        whether_abuse = "yes"
    elif re.search(r"\bnot\s+abused|no\s+abuse\b", low):
        whether_abuse = "no"

    conflict_type = ""
    if re.search(r"\b(manumission certificate|request(?:ing)?\s+to\s+be\s+manumitted|wished\s+to\s+be\s+manumitted|requests\s+repatriation|repatriation)\b", low):
        conflict_type = "manumission dispute"
    elif re.search(r"\bownership|belonged\s+to|present\s+owner|first\s+owner|second\s+owner\b", low):
        conflict_type = "ownership dispute"
    elif re.search(r"\bkidnapped|abducted\b", low):
        conflict_type = "kidnapping case"

    trial = ""
    if re.search(r"\brequests?\s+repatriation\b", low):
        trial = "repatriation requested"
    elif re.search(r"\brequest(?:ing)?\s+to\s+be\s+manumitted\b|\bgrant\s+.*manumission\s+certificate\b", low):
        trial = "manumission requested"
    elif re.search(r"\bwas\s+given\s+a\s+manumission\s+certificate\b|\bwere\s+given\s+manumission\s+certificates\b", low):
        trial = "manumission granted"
    elif re.search(r"\breleased\b", low):
        trial = "released"

    return {
        "Name": normalize_name(name),
        "Page": page,
        "Report Type": report_type,
        "Crime Type": crime_type,
        "Whether abuse": whether_abuse,
        "Conflict Type": conflict_type,
        "Trial": trial,
        "Amount paid": find_amount_paid(scoped),
    }


# ----------------------------- THREE-STAGE PIPELINE -----------------------------
def extract_people_events(ocr: str, doc_id: str, page: int, filename: str, *, deterministic_only: bool,
                          stats: Dict[str, int], logger: logging.Logger) -> CaseExtraction:
    page_type = classify_page(ocr)
    quality, _info = score_ocr_quality(ocr)
    document_date = extract_document_date(ocr)
    report_type = infer_report_type(ocr, stats, deterministic_only)

    if page_type in {"index", "record_metadata"}:
        return CaseExtraction(doc_id=doc_id, page=page, filename=filename, document_date=document_date,
                              people=[], report_type=report_type, page_type=page_type, status="skip_index_page")
    if quality == "garbled" and page_type not in {"declaration_summary", "telegram", "narrative_statement"}:
        return CaseExtraction(doc_id=doc_id, page=page, filename=filename, document_date=document_date,
                              people=[], report_type=report_type, page_type=page_type, status="skip_bad_ocr")

    doc_year = extract_year_from_document_date(document_date) or extract_year_from_document_date(ocr)
    named = deterministic_named_people(ocr)
    det_people: List[PersonCase] = []
    group_events = detect_group_destination_events(ocr, [n["name"] for n in named], doc_year)
    if named and re.search(r"\bThese\s+men\s+originally\s+lived\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\sas\s+slaves|$))", ocr, flags=re.I):
        m = re.search(r"\bThese\s+men\s+originally\s+lived\s+at\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|\sas\s+slaves|$))", ocr, flags=re.I)
        orig = normalize_place(m.group(1))
        if is_valid_place(orig):
            for n in [x["name"] for x in named]:
                group_events.setdefault(normalize_name(n), []).append(Event(type="presence", place=orig, evidence=sentence_around(ocr, m.start()), source="deterministic_group"))
    if named and re.search(r"\bthey\s+all\s+moved\s+to\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|where|$))", ocr, flags=re.I):
        m = re.search(r"\bthey\s+all\s+moved\s+to\s+([A-Za-z][A-Za-z'’\- ]{2,20}?)(?=\s*(?:,|\.|;|where|$))", ocr, flags=re.I)
        dest = normalize_place(m.group(1))
        if is_valid_place(dest):
            for n in [x["name"] for x in named]:
                group_events.setdefault(normalize_name(n), []).append(Event(type="presence", place=dest, evidence=sentence_around(ocr, m.start()), source="deterministic_group"))
    for item in named:
        name = normalize_name(item["name"])
        evs = extract_explicit_events_for_name(ocr, name, doc_year)
        evs.extend([Event(**asdict(ev)) for ev in group_events.get(name, [])])
        det_people.append(PersonCase(
            name=name,
            enslaved_status="explicit" if re.search(r"\b(slave|kidnapped|sold|refugee\s+slave|manumission)\b", item.get("evidence", ""), re.I) else "strong_inferred",
            enslaved_evidence=extract_name_evidence(ocr, name) or item.get("evidence", ""),
            events=evs,
        ))

    model_people: List[PersonCase] = []
    if not deterministic_only:
        try:
            model_people = model_extract_case(ocr, doc_id, stats)
        except Exception as e:
            logger.warning("Page %s model extraction failed: %s", page, e)

    return CaseExtraction(
        doc_id=doc_id,
        page=page,
        filename=filename,
        document_date=document_date,
        people=merge_people(det_people, model_people, doc_year),
        report_type=report_type,
        page_type=page_type,
        status="ok",
    )


def normalize_case_data(case_obj: CaseExtraction) -> CaseExtraction:
    doc_year = extract_year_from_document_date(case_obj.document_date)
    cleaned_people: List[PersonCase] = []
    for person in case_obj.people:
        name = normalize_name(person.name)
        if not is_likely_personal_name(name):
            continue
        evidence = normalize_ws(person.enslaved_evidence)[:300]
        if not evidence:
            continue
        events: List[Event] = []
        seen = set()
        for ev in person.events:
            nev = normalize_event(ev, doc_year)
            if nev is None:
                continue
            key = nev.canonical_key()
            if key in seen:
                continue
            seen.add(key)
            events.append(nev)
        events = stable_sort_events_by_date(events)
        cleaned_people.append(PersonCase(
            name=name,
            enslaved_status=coerce_enslaved_status(person.enslaved_status, evidence),
            enslaved_evidence=evidence,
            events=events,
        ))
    case_obj.people = cleaned_people
    if not case_obj.people and case_obj.status == "ok":
        case_obj.status = "skip_no_named_slave"
    return case_obj


def export_detail_rows(case_obj: CaseExtraction, current_ocr: str) -> List[Dict[str, Any]]:
    return [deterministic_meta(current_ocr, p.name, case_obj.page, case_obj.report_type) for p in case_obj.people]


def export_place_rows(case_obj: CaseExtraction) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for p in case_obj.people:
        timeline = events_to_place_timeline(p.events)
        if not timeline:
            rows.append({
                "Name": p.name,
                "Page": case_obj.page,
                "Place": "",
                "Order": "",
                "Arrival Date": "",
                "Date Confidence": "",
                "Time Info": "",
            })
            continue
        for idx, (place, arrival_date, date_confidence, time_text) in enumerate(timeline, 1):
            rows.append({
                "Name": p.name,
                "Page": case_obj.page,
                "Place": place,
                "Order": idx,
                "Arrival Date": arrival_date or "",
                "Date Confidence": date_confidence if arrival_date else "",
                "Time Info": time_text or "",
            })
    return rows


def export_canonical_json(case_obj: CaseExtraction) -> Dict[str, Any]:
    return {
        "doc_id": case_obj.doc_id,
        "page": case_obj.page,
        "filename": case_obj.filename,
        "document_date": case_obj.document_date,
        "report_type": case_obj.report_type,
        "page_type": case_obj.page_type,
        "status": case_obj.status,
        "people": [
            {
                "name": p.name,
                "enslaved_status": p.enslaved_status,
                "enslaved_evidence": p.enslaved_evidence,
                "events": [asdict(ev) for ev in p.events],
            }
            for p in case_obj.people
        ],
    }


# ----------------------------- IO / LOGGING -----------------------------
def write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in columns})
    os.replace(tmp, path)


def write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ner_extract_hybrid")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def assert_integrity() -> None:
    required = [
        "extract_people_events",
        "normalize_case_data",
        "export_detail_rows",
        "export_place_rows",
        "merge_people",
        "normalize_event",
        "events_to_place_timeline",
        "deterministic_meta",
    ]
    missing = [name for name in required if name not in globals() or not callable(globals()[name])]
    if missing:
        raise RuntimeError(f"Critical helpers missing at startup: {missing}")


# ----------------------------- MAIN -----------------------------
def process_file(path: pathlib.Path, args: argparse.Namespace, logger: logging.Logger,
                 stats_global: Dict[str, int]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    ocr = path.read_text(encoding="utf-8", errors="ignore")
    page_m = re.search(r"(\d+)", path.stem)
    page = int(page_m.group(1)) if page_m else 0
    stats = {"model_calls": 0, "extract_calls": 0, "repair_calls": 0}
    start = time.time()

    case_obj = extract_people_events(
        ocr=ocr,
        doc_id=path.stem,
        page=page,
        filename=path.name,
        deterministic_only=args.deterministic_only,
        stats=stats,
        logger=logger,
    )
    case_obj = normalize_case_data(case_obj)
    detail_rows = export_detail_rows(case_obj, ocr) if case_obj.status == "ok" else []
    place_rows = export_place_rows(case_obj) if case_obj.status == "ok" else []

    elapsed = round(time.time() - start, 2)
    stats_global["model_calls"] += stats["model_calls"]
    status_row = {
        "Page": page,
        "Filename": path.name,
        "Status": case_obj.status,
        "Named": len(case_obj.people),
        "Model Calls": stats["model_calls"],
        "Extract Calls": stats["extract_calls"],
        "Repair Calls": stats["repair_calls"],
        "Elapsed Seconds": elapsed,
        "Report Type": case_obj.report_type,
        "Page Type": case_obj.page_type,
    }
    logger.info(
        "[page %s] status=%s named=%s model_calls=%s elapsed_seconds=%.2f",
        page, case_obj.status, len(case_obj.people), stats["model_calls"], elapsed,
    )

    if args.save_json:
        os.makedirs(args.json_out_dir, exist_ok=True)
        write_json(os.path.join(args.json_out_dir, f"{path.stem}.json"), export_canonical_json(case_obj))

    return detail_rows, place_rows, status_row


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Hybrid slavery/manumission OCR extractor")
    ap.add_argument("--in_dir", default="typed_english_text_glmocr")
    ap.add_argument("--out_dir", default="hybrid_csv_out")
    ap.add_argument("--json_out_dir", default="hybrid_json_out")
    ap.add_argument("--save_json", action="store_true")
    ap.add_argument("--deterministic_only", action="store_true", help="Disable Ollama and run only deterministic extraction.")
    ap.add_argument("--glob", default="*.txt", help="Input glob within in_dir.")
    ap.add_argument("--limit", type=int, default=0, help="Optional max number of files to process.")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    assert_integrity()
    ap = build_arg_parser()
    args = ap.parse_args(argv)

    in_dir = pathlib.Path(args.in_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(str(out_dir))

    files = sorted(in_dir.glob(args.glob))
    if args.limit > 0:
        files = files[:args.limit]
    if not files:
        logger.error("No input files matched %s in %s", args.glob, in_dir)
        return 1

    detail_columns = ["Name", "Page", "Report Type", "Crime Type", "Whether abuse", "Conflict Type", "Trial", "Amount paid"]
    place_columns = ["Name", "Page", "Place", "Order", "Arrival Date", "Date Confidence", "Time Info"]
    status_columns = ["Page", "Filename", "Status", "Named", "Model Calls", "Extract Calls", "Repair Calls", "Elapsed Seconds", "Report Type", "Page Type"]

    detail_rows_all: List[Dict[str, Any]] = []
    place_rows_all: List[Dict[str, Any]] = []
    status_rows_all: List[Dict[str, Any]] = []
    stats_global = {"model_calls": 0}

    for path in files:
        try:
            drows, prows, srow = process_file(path, args, logger, stats_global)
            detail_rows_all.extend(drows)
            place_rows_all.extend(prows)
            status_rows_all.append(srow)
            write_csv(str(out_dir / "Detailed info.csv"), detail_rows_all, detail_columns)
            write_csv(str(out_dir / "name place.csv"), place_rows_all, place_columns)
            write_csv(str(out_dir / "run_status.csv"), status_rows_all, status_columns)
        except Exception:
            logger.exception("Page %s failed", path.name)
            status_rows_all.append({
                "Page": int(re.search(r"(\d+)", path.stem).group(1)) if re.search(r"(\d+)", path.stem) else 0,
                "Filename": path.name,
                "Status": "error",
                "Named": 0,
                "Model Calls": 0,
                "Extract Calls": 0,
                "Repair Calls": 0,
                "Elapsed Seconds": 0,
                "Report Type": "",
                "Page Type": "",
            })
            write_csv(str(out_dir / "run_status.csv"), status_rows_all, status_columns)

    logger.info("Finished %s files; total model_calls=%s", len(files), stats_global["model_calls"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
