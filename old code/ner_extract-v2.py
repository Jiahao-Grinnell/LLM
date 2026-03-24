#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multipass OCR -> slave extraction pipeline for historical slavery/manumission pages.

Key design choices:
- Keep Ollama as the backend.
- Use MULTIPLE focused model calls per page instead of one overloaded prompt.
- Recover explicit named slaves even when the model under-extracts via deterministic regex fallback.
- Skip pages with no NAMED slaves.
- Save CSV outputs incrementally after EVERY processed page.
- Infer a single report type from the first page by default.

Outputs
-------
1) Detailed info.csv
   Columns: Name, Page, Report Type, Crime Type, Whether abuse, Conflict Type, Trial, Amount paid

2) name place.csv
   Columns: Name, Page, Place, Order, Arrival Date, Date Confidence

Status
------
run_status.csv includes per-page model call counts.
"""

import argparse
import csv
import datetime as dt
import json
import logging
import os
import pathlib
import re
import time
from collections import defaultdict
from difflib import SequenceMatcher
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, ReadTimeout

# ------------------- OLLAMA CONFIG -------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:14b")
REQUEST_TIMEOUT = (10, 600)
MAX_CALL_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1200"))
DEFAULT_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX")
DEFAULT_NUM_CTX = int(DEFAULT_NUM_CTX) if DEFAULT_NUM_CTX and DEFAULT_NUM_CTX.isdigit() else None

_OLLAMA_SESSION = requests.Session()
_OLLAMA_SESSION.headers.update({"Connection": "keep-alive"})

_INDEX_LINE_PAT = re.compile(r"^\s*(?:\(?\d{1,4}\)?[\).]|[A-Z]\.|[ivxlcdm]{1,6}[\).])\s+", re.I)
_INDEX_LIST_HINTS = (
    "individual cases",
    "correspondence re",
    "manumission of slave",
    "manumission of slaves",
    "request by",
    "do -",
    "dv -",
)


def looks_like_index_page(ocr: str) -> bool:
    """Heuristic skip for list/index pages; leaves narrative pages unchanged."""
    if not ocr:
        return False
    lines = [ln.strip() for ln in ocr.splitlines() if ln.strip()]
    if len(lines) < 20:
        return False
    sample = lines[:120]
    enum = sum(1 for ln in sample if _INDEX_LINE_PAT.match(ln))
    short = sum(1 for ln in sample if len(ln) <= 120)
    low = "\n".join(sample).lower()
    hint_hits = sum(1 for h in _INDEX_LIST_HINTS if h in low)
    short_ratio = short / max(1, len(sample))
    return (enum >= 8 and short_ratio >= 0.55) or (hint_hits >= 2 and enum >= 5 and short_ratio >= 0.50)


def looks_like_record_metadata_page(ocr: str) -> bool:
    """Skip archive cover/about pages that mention cases but are not case pages."""
    if not ocr:
        return False
    head = normalize_ws((ocr or "")[:2600]).lower()
    hits = sum(
        1 for pat in [
            "this pdf was generated on",
            "qatar digital library",
            "the online record can be viewed at",
            "holding institution",
            "reference ior/",
            "about this record",
            "extent and format",
            "copyright for document",
            "open government licence",
        ]
        if pat in head
    )
    if hits >= 3:
        return True
    if "about this record" in head and "volume is comprised of correspondence" in head:
        return True
    return False


# ------------------- PLACE NORMALIZATION -------------------
PLACE_MAP = {
    # Gulf / Trucial Coast
    "shargah": "Sharjah",
    "sharjah": "Sharjah",
    "sharqah": "Sharjah",
    "sharjeh": "Sharjah",
    "shargal": "Sharjah",
    "abu dhabi": "Abu Dhabi",
    "abu bakara": "Abu Bakara",
    "abu baqarah": "Abu Bakara",
    "abu bakaran": "Abu Bakara",
    "murair": "Murair",
    "merai": "Murair",
    "sur": "Sur",
    "hemriyah": "Hamriyah",
    "hamriyah": "Hamriyah",
    "shindagha": "Shindagha",
    "shamdaghah": "Shindagha",
    "ras ul khaimah": "Ras al Khaimah",
    "ras ul khaima": "Ras al Khaimah",
    "ras al khaimah": "Ras al Khaimah",
    "rasul khaimah": "Ras al Khaimah",
    "umm al quwain": "Umm al Quwain",
    "umm ul quwain": "Umm al Quwain",
    "umm-ul-qaiwain": "Umm al Quwain",
    "umm-ul-quwain": "Umm al Quwain",
    "umm ul qaiwain": "Umm al Quwain",
    "umm al qaiwain": "Umm al Quwain",
    "umm el quwain": "Umm al Quwain",
    "jumairah": "Jumeirah",
    "jumair": "Jumeirah",
    "jumeirah": "Jumeirah",
    "ajmar": "Ajman",
    "aiman": "Ajman",
    "ajman": "Ajman",
    "debai": "Dubai",
    "dibai": "Dubai",
    "dobai": "Dubai",
    "dohai": "Dubai",
    "ebai": "Dubai",
    "dubai": "Dubai",
    "muscat": "Muscat",
    "mascat": "Muscat",
    "museat": "Muscat",
    "bahrein": "Bahrain",
    "bahrain": "Bahrain",
    "muharraq": "Muharraq",
    "henjam": "Henjam",
    "honjam": "Henjam",
    "bandar abbas": "Bandar Abbas",
    "lingah": "Lingah",
    "bushire": "Bushehr",
    "bushehr": "Bushehr",
    "busheir": "Bushehr",

    # Oman / inland
    "batinah": "Batinah",
    "batinah coast": "Batinah Coast",
    "khazrah of batinah": "Khazrah of Batinah",
    "khazrah of batina": "Khazrah of Batinah",
    "oman coast": "Oman Coast",
    "oman": "Oman",
    "badiya in oman": "Badiya, Oman",
    "amahn": "Oman",
    "amahan": "Oman",
    "amanan": "Oman",
    "jask": "Jask",
    "jiddeh": "Jeddah",
    "jiddah": "Jeddah",
    "mecca": "Mecca",
    "qatar": "Qatar",
    "eastern rufa": "Eastern Rufa",
    "eastern rufa'": "Eastern Rufa",

    # Persia / Baluchistan / East Africa
    "persia": "Persia",
    "mekran": "Mekran",
    "mokran": "Mekran",
    "minab": "Minab",
    "mirab": "Minab",
    "kuhru": "Kuhru",
    "kahru": "Kuhru",
    "gibrik": "Gibrik",
    "kuhistak": "Kuhistak",
    "seri": "Seri",
    "khan": "Khan",
    "rudbar": "Rudbar",
    "suza": "Suza",
    "bint": "Bint",
    "boshakird": "Boshakird",
    "boashakird": "Boshakird",
    "bashakird": "Boshakird",
    "bishakird": "Boshakird",
    "zanzibar": "Zanzibar",
    "selali": "Selali",
    "midi": "Midi",
    "abisinia": "Abyssinia",
    "abyssinia": "Abyssinia",
    "nejd": "Nejd",
    "naburi": "Naburi",
}
NON_GEO_PLACEHOLDERS = {
    "unknown", "unclear", "n/a", "na", "none", "nil", "there", "here",
    "residency", "political resident", "resident", "agency", "residency agency",
    "at sea", "diving grounds", "diving place", "this agency", "the agency", "africa",
}

MAX_PLACE_WORDS = 6
BAD_PLACE_TOKEN_PAT = re.compile(
    r"\b(and|to|from|therefore|told|landed|kept|captured|bought|sold|sent|took|arrived|reached|"
    r"earn|livelihood|godown|days|later|recorded|aged|statement|made)\b",
    re.I,
)
MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}
WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}

ROLE_TITLE_PAT = re.compile(
    r"\b(residency agent|political resident|treasury accounts officer|chief|shaikh|sheikh|major|"
    r"captain|lieutenant|lt\.?|mr\.?|mrs\.?|miss|nakhuda)\b",
    re.I,
)

DETAIL_COLUMNS = [
    "Name", "Page", "Report Type", "Crime Type", "Whether abuse", "Conflict Type", "Trial", "Amount paid"
]
PLACE_COLUMNS = ["Name", "Page", "Place", "Order", "Arrival Date", "Date Confidence", "Time Info"]
STATUS_COLUMNS = [
    "page", "filename", "status", "named_slaves", "detail_rows", "place_rows",
    "model_calls", "extract_calls", "repair_calls", "elapsed_seconds", "note"
]

# ------------------- PROMPTS -------------------
NAME_PASS_PROMPT = """You are extracting ONLY NAMED enslaved persons from one OCR page.

Return JSON only:
{{
  "named_slaves": [
    {{"name": string, "evidence": string}}
  ]
}}

Rules:
- Include ONLY people with a real personal name.
- Omit unnamed people entirely.
- Include a person only if the page explicitly states or strongly implies they are enslaved / a slave / kidnapped into slavery / sold / bought / manumitted / re-enslaved.
- Strong cues include phrases like:
  * a slave named X
  * statement made by X in a slave/manumission case
  * statement of slave X
  * slave girl X / slave boy X
  * grant X a manumission certificate
  * refugee slaves namely X, Y, Z
  * Slave No. 1 = X / Slave No. 2 - Y
- Exclude masters, buyers, heirs, rulers, sheikhs, officials, witnesses.
- Do NOT include a name merely because the page says "sold to X" or "a woman/man named X" unless X is clearly the enslaved/manumission subject.
- evidence must be a short quote <=25 words.
- Output JSON only, no markdown.

OCR TEXT:
<<<{ocr}>>>
"""

NAME_RECALL_PROMPT = """Extract EVERY NAMED enslaved person from this OCR page.

This is a recall-focused second pass.
Look especially for FULL names with connectors like bin / bint / al and lineage forms like daughter of / son of.
Look especially for:
- "a slave ... named X"
- "grant X a manumission certificate"
- "statement made by X" when the page is clearly about the enslavement/manumission of X
- "statement of slave X"
- "slave X who sought refuge"
- "refugee slaves namely X, Y, Z"
- numbered lists such as "Slave No. 1 = X" or "(1) X"
- lists introduced by "for delivery to:" or "the following refugee slaves"

Return JSON only:
{{
  "named_slaves": [
    {{"name": string, "evidence": string}}
  ]
}}

Important:
- Include ONLY people with real names.
- Do NOT include unnamed slaves.
- Do NOT include owners or officials.
- Do NOT include buyers/sellers who appear only in phrases like "sold to X".

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_PASS_PROMPT = """You are extracting the FULL ordered place timeline for one named slave from one OCR page.

Target slave name: {name}

Return JSON only:
{{
  "name": "{name}",
  "places": [
    {{
      "place": string,
      "order": integer,
      "arrival_date": string|null,
      "date_confidence": "explicit"|"derived_from_doc"|"unknown",
      "time_text": string|null,
      "evidence": string
    }}
  ]
}}

Rules:
- Preserve the FULL target name exactly as given.
- Extract EVERY real named place explicitly linked to {name}.
- The OCR may include the current page plus selected continuation pages from the SAME case; use continuation pages only to enrich this same slave.
- Include route places such as kidnapped from, brought to, imported to, sent to, sold to a man of, sold to [person] at, landed at, shipped to, from there to, thence to, complained in, sought refuge in, protected at, sent on to, and recorded at.
- If the page says "this Agency" or "the Agency", use the page's named location if explicit in the header.
- Keep the chronological order from earliest known place to latest known place on the page.
- order = 1,2,3,... for a clear route sequence.
- order = 0 only if linked but sequence is uncertain.
- arrival_date should be ISO when possible; otherwise null.
- time_text should keep a short phrase like "when he was 8 years old", "remained 3 years", "Recorded on 21st November 1934", or "by 10th November 1934".
- evidence must be <=25 words.
- Do not invent places.
- Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_RECALL_PROMPT = """Extract any missed places for the named slave below.

Target slave name: {name}

Return JSON only:
{{
  "name": "{name}",
  "places": [
    {{
      "place": string,
      "order": integer,
      "arrival_date": string|null,
      "date_confidence": "explicit"|"derived_from_doc"|"unknown",
      "time_text": string|null,
      "evidence": string
    }}
  ]
}}

Focus on places often missed in OCR pages:
- places after lowercase OCR variants like Debai/ebai, Shargah, Boashakird
- places in first-person routes: kidnapped from X, brought me to Y, sent me to Z, sold me to a man of W, sold me to [person] at X
- refuge/protection locations such as protected at X, escaped to X, sought refuge at X
- passive route wording such as was sent to X, were sent to X, were to send to X
- refuge locations from page headers when the text says this Agency
- complaint / recorded / refuge locations later in the page

Preserve full names and output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

META_PASS_PROMPT = """You are extracting case metadata for one NAMED slave from one OCR page.

Target slave name: {name}
Report Type (fixed): {report_type}
Page: {page}

Return JSON only:
{{
  "name": "{name}",
  "page": {page},
  "report_type": "{report_type}",
  "crime_type": string|null,
  "whether_abuse": "yes"|"no"|"" ,
  "conflict_type": string|null,
  "trial": string|null,
  "amount_paid": string|null,
  "evidence": string|null
}}

Rules:
- report_type must equal the fixed value above.
- crime_type = type of crime attached against that slave (kidnapping, sale, illegal detention, trafficking, etc.) only if supported.
- whether_abuse: yes if explicitly abused; no if explicitly states no abuse; otherwise empty string.
- conflict_type: conflict involved in the case only if supported.
- trial: formal verdict / outcome for that person only if supported.
- amount_paid: amount paid for manumission or release only if supported.
- Leave unsupported text fields null.
- Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

JSON_REPAIR_PROMPT = """Fix the following so it is valid JSON only. Do not add new facts.
Target top-level shape:
{schema}

TEXT TO FIX:
<<<{bad}>>>
"""


# ------------------- UTILITIES -------------------

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _strip_accents(s: str) -> str:
    if not s:
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))


def normalize_name(raw: str) -> str:
    if not raw:
        return ""
    s = re.split(r"\|\s*role\s*:", raw, flags=re.I)[0]
    s = _strip_accents(s)
    s = normalize_ws(s)
    s = s.strip(" ,.;:[]{}\"'")
    s = re.sub(r"^(?:Mst|Mrs|Miss|Mr)\.?\s+", "", s, flags=re.I)
    s = re.sub(r"\bslave\s*no\.?\s*\d+\b", "", s, flags=re.I)
    s = re.sub(r"\bslave\s*no\.?\b.*$", "", s, flags=re.I)
    s = re.sub(r"\b(ibn)\b", "bin", s, flags=re.I)
    s = re.sub(r"\b(bint)\b", "bint", s, flags=re.I)
    s = re.sub(r"\b(al|el|ul)\b", lambda m: m.group(1).lower(), s, flags=re.I)
    s = re.sub(r"\b(bin)\b", "bin", s, flags=re.I)
    s = re.sub(r"\b(abu)\b", "Abu", s, flags=re.I)
    s = re.sub(r"\b(umm)\b", "Umm", s, flags=re.I)
    s = re.sub(r"\b(?:aged\s+about|aged|small\s+son|eldest\s+son|recorded\s+on)\b.*$", "", s, flags=re.I)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    tokens = []
    for tok in s.split():
        low = tok.lower()
        if low in {"bin", "bint", "al", "el", "ul", "ibn"}:
            tokens.append("bin" if low == "ibn" else low)
        elif low in {"abu", "umm"}:
            tokens.append(low.title())
        else:
            tokens.append(tok[:1].upper() + tok[1:])
    return normalize_ws(" ".join(tokens))


def _name_skeleton(name: str) -> str:
    n = normalize_name(name).lower()
    n = _strip_accents(n)
    toks = []
    for tok in re.findall(r"[a-z']+", n):
        if tok in {"bin", "bint", "daughter", "son", "of", "al", "el", "ul", "abu", "umm"}:
            toks.append(tok)
            continue
        tok = tok.replace("v", "f").replace("p", "f").replace("ph", "f")
        tok = tok.replace("oo", "u").replace("ee", "i")
        toks.append(tok)
    return " ".join(toks)


def names_maybe_same_person(a: str, b: str) -> bool:
    a_n = normalize_name(a)
    b_n = normalize_name(b)
    if not a_n or not b_n:
        return False
    if a_n.lower() == b_n.lower():
        return True

    a_toks = [t.lower() for t in a_n.split()]
    b_toks = [t.lower() for t in b_n.split()]
    short, long = (a_toks, b_toks) if len(a_toks) <= len(b_toks) else (b_toks, a_toks)
    if long[:len(short)] == short and len(short) >= 1:
        return True
    if a_toks and b_toks and a_toks[0] == b_toks[0] and (len(a_toks) == 1 or len(b_toks) == 1):
        return True

    a_core = [t for t in a_toks if t not in {"bin", "bint", "daughter", "son", "of", "al", "el", "ul"}]
    b_core = [t for t in b_toks if t not in {"bin", "bint", "daughter", "son", "of", "al", "el", "ul"}]
    if len(a_core) == len(b_core) and a_core:
        sims = [SequenceMatcher(None, x, y).ratio() for x, y in zip(a_core, b_core)]
        if sum(sims) / len(sims) >= 0.83:
            return True
    return _name_skeleton(a_n) == _name_skeleton(b_n)


def choose_preferred_name(*names: str) -> str:
    candidates = [normalize_name(n) for n in names if normalize_name(n)]
    if not candidates:
        return ""
    def score(n: str) -> tuple:
        toks = n.split()
        low = n.lower()
        lineage_bonus = 1 if re.search(r"\b(?:daughter|son)\s+of\b", low) else 0
        connector_bonus = sum(1 for t in toks if t.lower() in {"bin", "bint"})
        weird_penalty = sum(1 for t in toks if len(t) >= 5 and sum(ch.lower() in "aeiou" for ch in t) == 0)
        return (lineage_bonus, connector_bonus, len(toks), len(n), -weird_penalty)
    return sorted(candidates, key=score, reverse=True)[0]


def canonicalize_name_against_context(name: str, case_ocr: str) -> str:
    base = normalize_name(name)
    if not base or not case_ocr:
        return base
    candidates = []
    for item in deterministic_listed_names(case_ocr) + deterministic_named_slaves(case_ocr):
        nm = normalize_name(item.get("name", ""))
        if nm and names_maybe_same_person(base, nm):
            candidates.append(nm)
    if not candidates:
        return base
    low_context = _strip_accents(case_ocr.lower())
    ranked = sorted(
        set(candidates + [base]),
        key=lambda n: (low_context.count(_strip_accents(n.lower())), len(n.split()), len(n)),
        reverse=True,
    )
    return choose_preferred_name(ranked[0], base)

def is_likely_personal_name(name: str) -> bool:
    if not name:
        return False
    n = normalize_name(name)
    if len(n) < 2:
        return False
    low = n.lower()
    bad_exact = {
        "woman", "man", "boy", "girl", "slave", "unknown", "unnamed", "a woman", "the woman",
        "the man", "the boy", "the girl", "the slave", "refugee slave", "slave girl", "slave boy",
        "dated", "received", "telegram", "code", "approved",
    }
    if low in bad_exact:
        return False
    if ROLE_TITLE_PAT.search(n) and len(n.split()) <= 5:
        return False
    if re.fullmatch(r"[A-Z]\.?(?:\s+[A-Z]\.?)?", n):
        return False
    if re.search(r"\b(statement|recorded|political|agency|resident|consulate|memorandum|certificate|delivered|delivery)\b", low):
        return False
    toks = [t for t in re.split(r"\s+", n) if t]
    if not toks:
        return False
    if toks[0].lower() in {"at", "from", "to", "the", "a", "an", "and", "my", "these", "this", "where", "when", "now", "sir", "your", "look"}:
        return False
    low_tokens = {t.lower() for t in toks}
    if low_tokens & {"obedient", "servant", "assistant", "agency", "bahrain", "political", "resident", "secretary", "captain", "negroes", "look", "like", "dated", "received", "telegram", "code", "approved"}:
        return False
    if len(toks) == 1 and len(toks[0]) <= 2:
        return False
    if sum(1 for t in toks if re.search(r"[A-Za-z]", t)) == 0:
        return False
    return True



def normalize_place(raw: str) -> str:
    if not raw:
        return ""
    s = _strip_accents(normalize_ws(raw))
    s = s.strip(" ,.;:[]{}\"'")
    s = s.replace("—", "-")
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = re.sub(r"\bof\s+Negro\s+slave\s+parents\b.*$", "", s, flags=re.I)
    if not s:
        return ""

    agency_loc = re.search(
        r"(?:residency\s+agency|political\s+agency|british\s+government'?s?\s+residency\s+agency)[,\s]+([A-Za-z][A-Za-z'’\-/ ]{2,60})",
        s,
        flags=re.I,
    )
    if agency_loc:
        s = agency_loc.group(1)

    s = re.sub(r"^(?:the\s+)?(?:residency\s+agency|political\s+agency|british\s+government'?s?\s+residency\s+agency|the\s+agency|this\s+agency)\b[, ]*", "", s, flags=re.I)
    s = re.sub(r"^(?:the\s+)?(?:town|port|island|coast)\s+of\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:at|in|to|from|near|off)\s+", "", s, flags=re.I)
    s = re.split(
        r"\b(?:named|where|who|whom|which|when|that|while|during|after|before|because|for\s+the\s+purpose|for\s+delivery|recorded|signed|dated|having|leaving|left|tomorrow|yesterday|safely|some\s+months?\s+ago|about\s+\d+\s+(?:years?|months?|days?)\s+ago|\d+\s+(?:years?|months?|days?)\s+(?:ago|previously)|as\s+slaves?\s+to|being\s+sent|intended\s+to|wanted\s+to)\b",
        s,
        maxsplit=1,
        flags=re.I,
    )[0]
    s = re.sub(r"\b(?:man-of-war|boat|dhow|steamship|ship|vessel|show|sambuk|sambuks|jallibaut|boom)\b.*$", "", s, flags=re.I)
    s = normalize_ws(s.strip(" ,.;:"))
    if not s:
        return ""
    low_s = s.lower()
    if low_s in {"the residency agency", "residency agency", "political agency", "the agency", "this agency"}:
        return ""
    if re.match(r"^(?:the\s+)?house\s+of\b", low_s):
        return ""
    if re.match(r"^(?:ruler|chief|shaikh|sheikh|wali|residency\s+agent|political\s+agent)\s+of\b", low_s):
        return ""

    low = low_s.replace("-", " ")
    low = re.sub(r"\bras\s+ul\b", "ras al", low)
    low = re.sub(r"\bul\b", "al", low)
    low = re.sub(r"\bel\b", "al", low)
    low = normalize_ws(low)
    low = low.replace("mlphinstone", "elphinstone")
    low = low.replace("khnjam", "henjam")
    low = low.replace("qiighm", "qishm").replace("qighm", "qishm")
    low = low.replace("dibai", "dubai").replace("debai", "dubai")
    low = low.replace("shargah", "sharjah")

    mapped = PLACE_MAP.get(low)
    if mapped:
        return mapped
    if len(s) <= 3 and s.isupper():
        return s
    return " ".join([w[:1].upper() + w[1:] if w else w for w in low.split(" ")])


def is_valid_place(place: str) -> bool:
    if not place:
        return False
    p = normalize_ws(place)
    if not p:
        return False
    tokens = [t for t in re.split(r"\s+", p) if t]
    if len(tokens) > MAX_PLACE_WORDS:
        return False
    low = p.lower()
    if BAD_PLACE_TOKEN_PAT.search(low) and len(tokens) >= 4:
        return False
    if re.search(r"\d", p):
        return False
    if low in NON_GEO_PLACEHOLDERS or low in {"present owner", "first owner", "second owner"}:
        return False
    if re.match(r"^(?:the\s+)?house\s+of\b", low):
        return False
    if re.match(r"^(?:ruler|chief|shaikh|sheikh|wali|residency\s+agent|political\s+agent)\s+of\b", low):
        return False
    if low in {"residency agency", "political agency", "the residency agency", "this agency", "the agency"}:
        return False
    if re.search(r"\b(?:aged|years?|months?|days?|manumission|certificate|slave|master|boat|dhow|show|sambuk|jallibaut|quarter)\b", low) and len(tokens) >= 3:
        return False
    return True

def sentence_split(text: str) -> List[str]:
    text = text.replace("\r", "\n")
    parts = re.split(r"(?<=[\.!?])\s+|\n{2,}", text)
    out = [normalize_ws(p) for p in parts if normalize_ws(p)]
    return out


def sentence_around(text: str, idx: int, max_words: int = 25) -> str:
    sentences = sentence_split(text)
    if not sentences:
        return ""
    # approximate lookup by running char lengths
    running = 0
    chosen = sentences[0]
    for s in sentences:
        running += len(s) + 1
        if running >= idx:
            chosen = s
            break
    words = chosen.split()
    return " ".join(words[:max_words])


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    # direct try
    try:
        return json.loads(text)
    except Exception:
        pass
    # fenced code block
    m = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # outermost object/array
    start_positions = [pos for pos in [text.find("{"), text.find("[")] if pos != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                snippet = text[start:i+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    break
    return None


def parse_day_month(text: str) -> Optional[Tuple[int, int]]:
    if not text:
        return None
    s = normalize_ws(text.lower().replace(",", " "))
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+([a-z]+)\b", s)
    if not m:
        return None
    day = int(m.group(1))
    mon_name = m.group(3)
    for k, v in MONTHS.items():
        if mon_name.startswith(k[:3]):
            return day, v
    return None


def extract_doc_year(text: str) -> Optional[int]:
    if not text:
        return None
    m = re.search(r"\b(17|18|19|20)\d{2}\b", text)
    return int(m.group(0)) if m else None


def to_iso_date(date_str: str, doc_year: Optional[int] = None) -> Tuple[Optional[str], str]:
    if not date_str:
        return None, "unknown"
    s = normalize_ws(date_str)
    m_par = re.search(r"\(=\s*([^\)]+)\)", s)
    if m_par:
        inner = normalize_ws(m_par.group(1))
        iso, conf = to_iso_date(inner, doc_year)
        if iso and conf != "unknown":
            return iso, conf
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s, "explicit"
    m = re.search(r"(?:\bD/?\s*)?(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", s, flags=re.I)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = yy
        if yy < 100:
            base = doc_year if doc_year else 1900
            century = (base // 100) * 100
            year = century + yy
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{year:04d}-{mm:02d}-{dd:02d}", "explicit"
    m = re.search(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})\b", s)
    if m:
        mon_name, dd, yyyy = m.group(1), int(m.group(2)), int(m.group(3))
        for k, v in MONTHS.items():
            if mon_name.lower().startswith(k[:3]):
                return f"{yyyy:04d}-{v:02d}-{dd:02d}", "explicit"
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+)\s+(\d{4})\b", s)
    if m:
        dd, mon_name, yyyy = int(m.group(1)), m.group(2), int(m.group(3))
        for k, v in MONTHS.items():
            if mon_name.lower().startswith(k[:3]):
                return f"{yyyy:04d}-{v:02d}-{dd:02d}", "explicit"
    dm = parse_day_month(s)
    if dm and doc_year:
        dd, mm = dm
        return f"{doc_year:04d}-{mm:02d}-{dd:02d}", "derived_from_doc"
    return None, "unknown"


def extract_header_location(text: str) -> str:
    if not text:
        return ""
    pats = [
        r"\b([A-Z][A-Za-z'’\-/]+(?:\s+[A-Z][A-Za-z'’\-/]+){0,3}),\s*(?:\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"\bPolitical Agency[^\n]*\.\s*\n\s*([A-Z][A-Za-z'’\-/]+)",
        r"\bFrom\s*[-:]?\s*The Residency Agent,\s*([A-Z][A-Za-z'’\-/]+)",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.I)
        if m:
            loc = normalize_place(m.group(1))
            if is_valid_place(loc):
                return loc
    return ""


def extract_page_dates(text: str, doc_year: Optional[int]) -> List[Tuple[str, str]]:
    candidates = []
    patterns = [
        r"\(=\s*[^\)]+\)",
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
        r"\b[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\s+\d{4}\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            raw = m.group(0)
            window = text[max(0, m.start()-30):min(len(text), m.end()+30)]
            if re.search(r"IOR/R|Reference:|M\.S\.|Ms\.|No\.\s*\d+/?\d* of \d{4}", window, flags=re.I):
                continue
            iso, conf = to_iso_date(raw, doc_year)
            if iso:
                candidates.append((m.start(), iso, conf))
    candidates.sort(key=lambda x: x[0])
    seen = set(); uniq=[]
    for _, iso, conf in candidates:
        if iso in seen:
            continue
        seen.add(iso); uniq.append((iso, conf))
    return uniq


def find_amount_paid(text: str) -> Optional[str]:
    pats = [
        r"\bpaid\s+([0-9]+(?:\s+[A-Za-z]+)?)\b",
        r"\bfor\s+([0-9]+\s+rupees?)\b",
        r"\b([0-9]+\s+rupees?)\b",
        r"\b([0-9]+\s+dollars?)\b",
    ]
    for pat in pats:
        m = re.search(pat, text, flags=re.I)
        if m:
            return normalize_ws(m.group(1))
    return None


def infer_report_type_from_text(first_page_text: str) -> str:
    low = (first_page_text or "").lower()
    manumission_score = 0
    for pat in [
        r"manumission certificate", r"grant .* manumission certificate", r"manumitted",
        r"free man", r"freedom", r"certificate in support of .* freedom", r"re-enslave",
    ]:
        if re.search(pat, low):
            manumission_score += 1
    return "manumission" if manumission_score >= 1 else "correspondence"



def classify_document_page(text: str) -> str:
    """Lightweight page-type classifier used only for gating, not final output labels."""
    if not text or looks_like_index_page(text):
        return "index"
    if looks_like_record_metadata_page(text):
        return "record_metadata"
    low = (text or "").lower()

    if re.search(r"\b(statement\s+made\s+by|statement\s+of(?:\s+slave)?|i\s+was\s+born|i\s+was\s+kidnapped|i\s+was\s+brought|i\s+served|i\s+managed\s+to\s+escape)\b", low):
        return "narrative_statement"

    if re.search(r"\b(was\s+given\s+a\s+manumission\s+certificate|were\s+given\s+manumission\s+certificates|for\s+delivery\s+to\s*:|the\s+following\s+refugee\s+slaves|refugee\s+slaves\s+namely|slave\s*no\.?\s*\d+\s*[:=\-])\b", low):
        return "administrative_route"

    if re.search(r"\b(first\s+batch\s+of\s+slaves|second\s+batch\s+of\s+slaves|history\s+of\s+the\s+slaves|subsequent\s+investigation|representing\s+themselves\s+as\s+slaves|requesting\s+to\s+be\s+manumitted|wished\s+to\s+be\s+manumitted|originally\s+lived\s+at|they\s+all\s+moved\s+to)\b", low):
        return "administrative_route"

    if re.search(r"\b(from\s*[-:]?\s*the\s+residency\s+agent|political\s+agency|political\s+resident|consulate|memorandum|forward\s+herewith|submitted\s+for\s+information|copy\s+forwarded|attached\s+report)\b", low):
        return "administrative_memo"

    return "unknown"

def score_ocr_quality(text: str) -> Tuple[str, Dict[str, Any]]:
    """Return quality bucket and diagnostics.

    normal: readable enough for full extraction
    salvageable: noisy but has useful anchors
    garbled: mostly noise unless strong anchors exist
    """
    text = text or ""
    low = text.lower()
    tokens = re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)
    total_tokens = max(1, len(tokens))
    weird_tokens = sum(1 for t in tokens if len(t) >= 10 and not re.search(r"(?:statement|manumission|certificate|political|residency|agency|resident|bahrein|bahrain|shargah|sharjah|mekran|muscat|zanzibar)", t, re.I) and sum(ch.lower() in 'aeiou' for ch in t) <= 1)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    repeated_line_ratio = 0.0
    if lines:
        counts = {}
        for ln in lines:
            key = normalize_ws(ln.lower())
            counts[key] = counts.get(key, 0) + 1
        repeated = sum(v for v in counts.values() if v >= 3)
        repeated_line_ratio = repeated / max(1, len(lines))

    repeated_phrase_hits = 0
    for phrase in [
        "reason for its presence there",
        "on enquiring into the reason",
        "presence there and on enquiring",
    ]:
        if low.count(phrase) >= 2:
            repeated_phrase_hits += 1

    strong_anchor = bool(re.search(
        r"\b(statement\s+made\s+by|statement\s+of(?:\s+slave)?|slave\s*no\.?\s*\d+|grant\s+.+?manumission\s+certificate|refugee\s+slaves\s+namely|the\s+following\s+refugee\s+slaves|was\s+given\s+a\s+manumission\s+certificate|were\s+given\s+manumission\s+certificates)\b",
        low,
    ))

    short_alpha_lines = sum(1 for ln in lines[:80] if re.search(r"[A-Za-z]", ln) and len(re.findall(r"[A-Za-z]", ln)) < 12)
    line_count = max(1, len(lines[:80]))
    short_line_ratio = short_alpha_lines / line_count
    weird_ratio = weird_tokens / total_tokens

    info = {
        "strong_anchor": strong_anchor,
        "repeated_line_ratio": round(repeated_line_ratio, 3),
        "repeated_phrase_hits": repeated_phrase_hits,
        "weird_ratio": round(weird_ratio, 3),
        "short_line_ratio": round(short_line_ratio, 3),
    }

    if (repeated_phrase_hits >= 1 and repeated_line_ratio >= 0.12) or (weird_ratio >= 0.18 and short_line_ratio >= 0.45):
        return ("salvageable" if strong_anchor else "garbled"), info
    if weird_ratio >= 0.10 or repeated_line_ratio >= 0.08:
        return "salvageable", info
    return "normal", info


def page_supports_personal_route(page_type: str, text: str, ocr_quality: str) -> bool:
    low = (text or "").lower()
    if page_type == "narrative_statement":
        return True
    if page_type == "administrative_route":
        return True
    if page_type == "administrative_memo":
        return bool(re.search(r"\b(last\s+heard\s+of\s+at|was\s+given\s+a\s+manumission\s+certificate|were\s+given\s+manumission\s+certificates|from\s+[A-Z][A-Za-z'’\- ]+\s+to\s+[A-Z][A-Za-z'’\- ]+|applying\s+for\s+the\s+grant\s+of\s+manumission|sought\s+refuge|took\s+refuge|recorded\s+at)\b", text, re.I))
    if page_type == "unknown":
        if ocr_quality == "garbled":
            return False
        return bool(re.search(r"\b(i\s+was\s+born|i\s+was\s+kidnapped|brought\s+me\s+to|sent\s+me\s+to|sold\s+me\s+to|sought\s+refuge|protected\s+at|recorded\s+at|manumission\s+certificate|born\s+at|born\s+in|native\s+of|shipped\s+from|landed\s+at|sold\s+at|moved\s+to|originally\s+lived\s+at)\b", low))
    return False


def blank_place_row(name: str, page: int) -> Dict[str, Any]:
    return {
        "Name": normalize_name(name),
        "Page": page,
        "Place": "",
        "Order": "",
        "Arrival Date": "",
        "Date Confidence": "",
        "Time Info": "",
    }


def _setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("ner_extract_multipass")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def write_csv(path: str, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for row in rows:
            safe = {k: row.get(k, "") for k in columns}
            w.writerow(safe)
    os.replace(tmp, path)


def write_state(log_dir: str, state: Dict[str, Any]) -> None:
    tmp = os.path.join(log_dir, "run_state.json.tmp")
    path = os.path.join(log_dir, "run_state.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


# ------------------- OLLAMA -------------------
def call_ollama(prompt: str, stats: Dict[str, int], *, num_predict: int = DEFAULT_NUM_PREDICT,
                num_ctx: Optional[int] = DEFAULT_NUM_CTX) -> str:
    last_err = None
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
            r = _OLLAMA_SESSION.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return (r.json().get("response") or "").strip()
        except (ReadTimeout, ConnectionError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            time.sleep(wait)
        except requests.HTTPError as e:
            last_err = e
            if attempt < MAX_CALL_RETRIES:
                time.sleep(2 * attempt)
            else:
                break
    raise RuntimeError(f"Ollama call failed after retries: {last_err}")


def call_json_prompt(prompt: str, stats: Dict[str, int], schema_hint: str,
                     *, num_predict: int = DEFAULT_NUM_PREDICT) -> Optional[Any]:
    raw = call_ollama(prompt, stats, num_predict=num_predict)
    parsed = extract_json(raw)
    if parsed is not None:
        stats["extract_calls"] += 1
        return parsed
    repaired = call_ollama(JSON_REPAIR_PROMPT.format(schema=schema_hint, bad=raw), stats, num_predict=800)
    stats["repair_calls"] += 1
    parsed = extract_json(repaired)
    if parsed is not None:
        return parsed
    return None


# ------------------- DETERMINISTIC FALLBACKS -------------------
SLAVERY_CONTEXT_PAT = re.compile(
    r"\b(slave|enslave|enslaved|manumission|manumitted|free man|freedom|kidnapped|abducted|sold|bought|"
    r"purchased|re-enslave|re enslave|took refuge|manumission certificate)\b",
    re.I,
)

NAME_WORD = r"[A-Z][A-Za-z'’\-]+"
NAME_CONNECTOR = r"(?:bin|ibn|bint|al|el|ul|Abu|Umm)"
NAME_SEQ = rf"{NAME_WORD}(?:\s+(?:{NAME_WORD}|{NAME_CONNECTOR})){{0,6}}"
NAME_WITH_LINEAGE = rf"((?:Mst\.?\s+|Mrs\.?\s+|Miss\s+|Mr\.?\s+)?{NAME_SEQ}(?:\s*,?\s*(?:daughter|son)\s+of\s+{NAME_SEQ})?)"
FULL_NAME_ONLY_PAT = re.compile(rf"^(?:Mst\.?\s+|Mrs\.?\s+|Miss\s+|Mr\.?\s+)?{NAME_SEQ}(?:\s*,?\s*(?:daughter|son)\s+of\s+{NAME_SEQ})?$", re.I)
LIST_NAME_PAT = re.compile(NAME_WITH_LINEAGE, re.I)

NAMED_PATTERNS = [
    re.compile(rf"(?i:\ba slave(?:\s+of\s+[A-Z][A-Za-z'’\-]+)?\s+named\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bdomestic slave\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bmanumission of(?: the)?(?: domestic)? slave\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bstatement\s+made\s+by(?:\s+the)?(?:\s+slave)?(?:\s+girl|\s+boy|\s+woman)?\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bstatement\s+of(?:\s+slave)?\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bgrant\s+)" + NAME_WITH_LINEAGE + r"(?i:\s+a\s+manumission\s+certificate\b)"),
    re.compile(rf"(?i:\bregarding\s+slave\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bslave\s+)" + NAME_WITH_LINEAGE + r"(?i:\s+who\s+sought\s+refuge)"),
    re.compile(rf"(?i:\bslave\s+girl[,\s]+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bslave\s+boy[,\s]+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bmanumitted\s+slave\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\b(?:a|the)\s+Baluchi\s+(?:boy|girl)\s+named\s+)" + NAME_WITH_LINEAGE + r"\b"),
    re.compile(rf"(?i:\bslave\s+)" + NAME_WITH_LINEAGE + r"(?i:[,\.]|\s+is\b|\s+claiming\b|\s+took\b|\s+of\b|\s+aged\b)"),
]

LIST_NAME_SEG_PAT = re.compile(r"\b(?:refugee\s+slaves?|slaves?)\s+namely\s+(.{0,320})", re.I)
LIST_CLEAN_PAT = re.compile(r"\b(?:one|two|three|four|five|six|seven|eight)\s+(?:swahilis?|baluchis?|baluch|slaves?|boys?|girls?|women|men)\b", re.I)


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
    piece = re.sub(r"^No\.?\s*\d+\s*[.:-]?\s*", "", piece, flags=re.I)
    piece = re.sub(r"^slave\s*no\.?\s*\d+\s*[:=\-]\s*", "", piece, flags=re.I)
    piece = re.sub(r"\([^)]*\)", "", piece)
    piece = re.sub(r"\b(?:aged\s+about|born\s+at|recorded\s+on|eldest\s+son|small\s+son)\b.*$", "", piece, flags=re.I)
    piece = re.sub(r"\b(?:are|is|was|were|who|requests?|requesting|being|forwarded|arriving|accompanies|please|kindly)\b.*$", "", piece, flags=re.I)
    piece = piece.strip(" ,.;:")
    if FULL_NAME_ONLY_PAT.match(piece):
        return normalize_name(piece)
    return ""


def deterministic_listed_names(ocr: str) -> List[Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    text = ocr or ""

    for m in LIST_NAME_SEG_PAT.finditer(text):
        seg = re.split(r"[.;:\n]", m.group(1))[0]
        seg = LIST_CLEAN_PAT.sub(" ", seg)
        seg = seg.replace(" and ", ", ")
        for piece in [p.strip() for p in seg.split(',') if p.strip()]:
            nm = _extract_name_from_piece(piece)
            if nm:
                _add_name_candidate(out, nm, sentence_around(text, m.start()) or seg)

    for m in re.finditer(rf"\bslave\s*no\.?\s*\d+\s*[:=\-]\s*" + NAME_WITH_LINEAGE, text, flags=re.I):
        nm = normalize_name(m.group(1))
        if nm:
            _add_name_candidate(out, nm, sentence_around(text, m.start()))

    for m in re.finditer(rf"(?m)^\s*No\.?\s*\d+\s*[.:=-]\s*" + NAME_WITH_LINEAGE + r"\b", text):
        nm = normalize_name(m.group(1))
        if nm:
            _add_name_candidate(out, nm, sentence_around(text, m.start()) or normalize_ws(m.group(0)))

    for m in re.finditer(rf"(?i:\b(?:\d+\s+)?slave(?:\s+young\s+[A-Za-z-]+)?\s+named\s+)" + NAME_WITH_LINEAGE + r"\b", text):
        nm = normalize_name(m.group(1))
        if nm:
            _add_name_candidate(out, nm, sentence_around(text, m.start()) or normalize_ws(m.group(0)))

    seg_pats = [
        r"forward\s+herewith[^.\n]{0,220}?\bby\s+(.{0,260})",
        r"made\s+before\s+me\s+by(?:\s+the\s+refugee\s+slaves,)?\s*(.{0,260})",
        r"for\s+delivery\s+to\s*:?\s*(.{0,260})",
        r"to\s+say\s+that\s+(.{0,180}?)\s+were\s+given\s+Manumission\s+Certificates",
        r"the\s+following\s+refugee\s+slaves[^.\n]{0,120}?:\s*(.{0,260})",
        r"forwarded\s+herewith\s+for\s+favour\s+of\s+delivery\s+to\s+(.{0,220})",
    ]
    for pat in seg_pats:
        for m in re.finditer(pat, text, flags=re.I | re.S):
            seg = normalize_ws(m.group(1))
            seg = re.split(r"\b(?:Please|Kindly|U\.E\.|By order|Captain|Secretary|Political Agent)\b", seg, maxsplit=1, flags=re.I)[0]
            seg = seg.replace(" and ", ", ")
            seg = re.sub(r"\(\d+\)", ", ", seg)
            for piece in [p.strip() for p in seg.split(',') if p.strip()]:
                nm = _extract_name_from_piece(piece)
                if nm:
                    _add_name_candidate(out, nm, sentence_around(text, m.start()) or seg)

    list_heading = re.search(r"(?:the\s+following\s+refugee\s+slaves[^.]*\.|First\s+batch\s+of\s+slaves|Second\s+batch\s+of\s+slaves|the\s+persons\s+concerned\s+are)\s*[:.]?", text, flags=re.I)
    if list_heading:
        block = text[list_heading.end():list_heading.end()+500]
        for line in block.splitlines():
            line_n = normalize_ws(line)
            if not line_n:
                continue
            if re.search(r"\b(dated|memorandum|letter|political agent|secretary|captain|please|kindly|history|investigation)\b", line_n, flags=re.I):
                continue
            nm = _extract_name_from_piece(line_n)
            if nm:
                _add_name_candidate(out, nm, line_n)

    return list(out.values())


def deterministic_named_slaves(ocr: str) -> List[Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    if not ocr or not SLAVERY_CONTEXT_PAT.search(ocr):
        return []
    for pat in NAMED_PATTERNS:
        for m in pat.finditer(ocr):
            raw = m.group(m.lastindex) if m.lastindex else m.group(0)
            evidence = sentence_around(ocr, m.start()) or " ".join(ocr[max(0, m.start()-80):m.end()+80].split()[:25])
            if re.search(r"\b(?:sold\s+me\s+to|sold\s+him\s+to|sold\s+her\s+to|bought\s+by|purchased\s+by|my\s+master(?:'s)?\s+name\s+is)\b", evidence, flags=re.I):
                if not re.search(r"\b(statement\s+(?:made\s+by|of)|grant\s+.+?manumission\s+certificate|slave\s+girl|slave\s+boy|manumitted\s+slave)\b", evidence, flags=re.I):
                    continue
            _add_name_candidate(out, raw, evidence)
    for item in deterministic_listed_names(ocr):
        _add_name_candidate(out, item["name"], item["evidence"])
    return list(out.values())


PLACE_CAPTURE = r"([A-Za-z][A-Za-z'’\-/]+(?:\s+(?:of|al|ul|el|in|the|[A-Za-z][A-Za-z'’\-/]+)){0,5})"
PLACE_REGEXES = [
    (re.compile(rf"(?i:\bkidnapped\s+(?:me|him|her)?\s*from\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bcaptured\s+(?:me|him|her)?\s*from\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bimported\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bbrought\s+(?:me|him|her)?(?:\s+by\s+land(?:\s+route)?)?\s*to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bsent\s+(?:me|him|her)?\s*to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\b(?:was|were)\s+sent\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bwere\s+to\s+send\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\btaken\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\btaken\b[^\n\.;:]{0,80}?\bto\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bthence(?:\s+shipped\s+me\s+in\s+a\s+boat\s+and\s+landed\s+me\s+at|\s+to|\s+landed\s+me\s+at)\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bfrom\s+there\s+(?:he\s+)?(?:took|brought|sent)\s+(?:me|him|her)?(?:\s+in\s+a\s+boat)?\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bfrom\s+there\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bsailed\s+from\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\blanded\s+(?:me|him|her)?\s+at\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bembark(?:ed)?\s+(?:me|him|her)?(?:\s+in\s+a\s+boat)?\s+and\s+brought\s+(?:me|him|her)?\s+to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bsold\s+(?:me|him|her)?\s*to\s+(?:one|a\s+man|man)\s+of\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bsold\s+(?:me|him|her)?\s*to\s+[^\n\.\,;:]{0,80}?\bat\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bsold\s+(?:me|him|her)?\s*to\s+[^\n\.\,;:]{0,80}?\bin\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bprotected\s+by\s+[^\n\.\,;:]{0,80}?\bat\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\b(?:escaped|escape(?:d)?)\s+(?:with\s+[^\n\.]{0,40}?\s+)?to\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bcomplained(?:\s+twice)?\s+to\s+.*?\b(?:Agent|Chief),?\s*)" + PLACE_CAPTURE), "assoc"),
    (re.compile(rf"(?i:\brecorded\s+at\s+(?:the\s+Political\s+Agency,?\s*)?)" + PLACE_CAPTURE), "assoc"),
    (re.compile(rf"(?i:\bborn\s+at\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bborn\s+in\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bnative\s+of\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bshipped\s+from\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bbirth\s+place\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bmy\s+town\s+is\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bhome\s+is\s+in\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\bnative\s+place\s+)" + PLACE_CAPTURE), "route"),
    (re.compile(rf"(?i:\boriginally\s+from\s+)" + PLACE_CAPTURE), "route"),
]
DATE_PAT = re.compile(r"(?:\(=\s*[^\)]+\)|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b[A-Z][a-z]+\s+\d{1,2},\s*\d{4}\b|\b\d{1,2}(?:st|nd|rd|th)?\s+[A-Z][a-z]+(?:\s+\d{4})?\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b)")
TIME_TEXT_PAT = re.compile(r"\b(?:when I was[^,.]*|when he was[^,.]*|when she was[^,.]*|aged about[^,.]*|remained with him for[^,.]*|after some years|Two years after[^,.]*|for about one month|for about \d+ years?|about \d+ years? ago|Recorded on[^,.]*|Dated[^,.]*|in the month of [^,.]*|during the summer of last year|at the end of this year's diving season|by \d{1,2}(?:st|nd|rd|th)? [A-Z][a-z]+ \d{4}|[A-Z][a-z]+ \d{4})", re.I)



def _word_or_digit_to_int(token: str) -> Optional[int]:
    if not token:
        return None
    token = token.lower().strip()
    if token.isdigit():
        return int(token)
    return WORD_NUM.get(token)


def extract_anchor_date(text: str, doc_year: Optional[int]) -> Optional[dt.date]:
    page_dates = extract_page_dates(text or "", doc_year)
    if not page_dates:
        return None
    iso = page_dates[0][0]
    try:
        return dt.date.fromisoformat(iso)
    except Exception:
        return None


def derive_relative_arrival(snippet: str, doc_year: Optional[int], anchor_date: Optional[dt.date] = None) -> Tuple[str, str]:
    if not snippet:
        return "", ""
    low = normalize_ws(snippet).lower()
    if not doc_year and anchor_date:
        doc_year = anchor_date.year

    if anchor_date:
        m = re.search(r"\b(?:about\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+days?\s+(?:ago|previously)\b", low)
        if m:
            n = _word_or_digit_to_int(m.group(1))
            if n:
                return (anchor_date - dt.timedelta(days=n)).isoformat(), "derived_from_doc"

    m = re.search(r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+or\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", low)
    if m and doc_year:
        a = _word_or_digit_to_int(m.group(1))
        b = _word_or_digit_to_int(m.group(2))
        if a and b:
            return str(doc_year - max(a, b)), "derived_from_doc"

    m = re.search(r"\b(?:about\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+years?\s+ago\b", low)
    if m and doc_year:
        n = _word_or_digit_to_int(m.group(1))
        if n:
            return str(doc_year - n), "derived_from_doc"

    m = re.search(r"\b(?:about\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+months?\s+ago\b", low)
    if m and (anchor_date or doc_year):
        n = _word_or_digit_to_int(m.group(1))
        if n:
            base = anchor_date or dt.date(doc_year, 12, 31)
            year = base.year
            month = base.month - n
            while month <= 0:
                month += 12
                year -= 1
            return f"{year:04d}-{month:02d}", "derived_from_doc"

    m = re.search(r"\baged\s+about\s+(\d+)\s+years?\b", low)
    if m and doc_year:
        return str(doc_year - int(m.group(1))), "derived_from_doc"

    if "some months ago" in low and (anchor_date or doc_year):
        base = anchor_date or dt.date(doc_year, 12, 31)
        year = base.year if base.month > 3 else base.year - 1
        month = max(1, base.month - 3)
        return f"{year:04d}-{month:02d}", "derived_from_doc"

    return "", ""


def extract_nearby_date(snippet: str, doc_year: Optional[int], anchor_date: Optional[dt.date] = None) -> Tuple[str, str]:
    if not snippet:
        return "", ""
    m = DATE_PAT.search(snippet)
    if m:
        iso, conf = to_iso_date(m.group(0), doc_year=doc_year)
        return iso or "", conf or ""
    return derive_relative_arrival(snippet, doc_year, anchor_date)


def extract_time_text(snippet: str) -> str:
    if not snippet:
        return ""
    m = TIME_TEXT_PAT.search(snippet)
    if m:
        return normalize_ws(m.group(0))
    rel = re.search(r"\b(?:some\s+months?\s+ago|about\s+\d+\s+(?:years?|months?|days?)\s+ago|\d+\s+(?:years?|months?|days?)\s+(?:ago|previously)|\d+\s+or\s+\d+\s+years?\s+ago|aged\s+about\s+\d+\s+years?)\b", snippet, flags=re.I)
    return normalize_ws(rel.group(0)) if rel else ""


def evidence_disqualifies_place(evidence: str, place: str, header_loc: str = "") -> bool:
    ev = normalize_ws(evidence).lower()
    p = normalize_place(place).lower()
    if not p:
        return True
    if p in {"residency agency", "political agency", "the residency agency", "this agency", "the agency"}:
        return True

    if any(re.search(pat, ev) for pat in [
        r"\b(?:being|were|was)\s+sent\s+to\b",
        r"\bleaving\s+(?:the\s+following\s+day|tomorrow)?\s*for\b",
        r"\b(?:intended|wanted)\s+to\s+sell\b",
        r"\bto\s+be\s+sold\s+to\b",
        r"\bwish\s+to\s+stay\b",
    ]):
        if not re.search(r"\b(arrived|reached|landed|took refuge|escaped to|came to|brought me to|took me to|sold .* in|sold .* at)\b", ev):
            return True

    if re.search(r"\b(?:ruler|chief|shaikh|sheikh|wali)\s+of\s+" + re.escape(p) + r"\b", ev):
        return True
    if header_loc and p == header_loc.lower() and re.search(r"\b(?:dated|from|to)\b", ev) and not re.search(r"\b(arrived|reached|took refuge|recorded|born|lived|moved)\b", ev):
        return True
    return False


def _maybe_add_place(found, pos, raw_place, order_kind, evidence, doc_year, anchor_date=None, default_page_date=("", ""), header_loc=""):
    place = normalize_place(raw_place)
    if not is_valid_place(place):
        return
    if evidence_disqualifies_place(evidence, place, header_loc):
        return
    dt_val, dt_conf = extract_nearby_date(evidence, doc_year, anchor_date)
    if not dt_val and re.search(r"\b(?:arrived|reached|took refuge|recorded|anchored at|landed at)\b", evidence, flags=re.I):
        dt_val, dt_conf = default_page_date
    time_text = extract_time_text(evidence)
    if not dt_val:
        dt_conf = ""
    found.append((pos, place, 1 if order_kind == "route" else 0, dt_val, dt_conf, time_text, evidence))




def extract_focus_text_for_name(ocr: str, name: str) -> str:
    if not ocr or not name:
        return ocr
    low = _strip_accents(ocr.lower())
    target = normalize_name(name)
    aliases = [a for a in _name_aliases(target) if a]

    # Prefer numbered declaration blocks like "No.4. NAME ..." and stop at the next numbered block.
    block_starts = list(re.finditer(r"(?im)^\s*No\.?\s*\d+\s*[.:=-]?\s*(.*)$", ocr))
    if block_starts:
        alias_lows = [_strip_accents(a.lower()) for a in aliases]
        for i, m in enumerate(block_starts):
            line = _strip_accents(normalize_ws(m.group(1)).lower())
            if any(a and re.search(rf"\b{re.escape(a)}\b", line) for a in alias_lows):
                start = m.start()
                end = block_starts[i + 1].start() if i + 1 < len(block_starts) else len(ocr)
                return ocr[start:end]

    probes = [a.lower() for a in aliases if a] + [target.lower(), target.lower().split()[0]]
    positions = [low.find(_strip_accents(p)) for p in probes if p and low.find(_strip_accents(p)) != -1]
    if not positions:
        return ocr
    anchor = min(positions)
    start = max(0, anchor - 350)
    end = min(len(ocr), anchor + 4200)
    tail = ocr[start:end]
    rel_anchor = anchor - start
    cut_points = []
    for marker in [r"\bStatement\s+made\s+by\b", r"\bStatement\s+of\b", r"\bFrom\s*[-:]?\s*The Residency Agent\b", r"(?im)^\s*No\.?\s*\d+\s*[.:=-]?", r"\bNo\.\s*\d+\s+of\s+\d{4}\b"]:
        for mm in re.finditer(marker, tail, flags=re.I):
            if mm.start() > rel_anchor + 120:
                cut_points.append(mm.start())
    if cut_points:
        tail = tail[:min(cut_points)]
    return tail


def deterministic_places_for_page(ocr: str, name: str, doc_year: Optional[int]) -> List[Dict[str, Any]]:
    header_loc = extract_header_location(ocr)
    page_dates = extract_page_dates(ocr, doc_year)
    default_page_date = page_dates[0] if page_dates else ("", "")
    anchor_date = extract_anchor_date(ocr, doc_year)
    focus = extract_focus_text_for_name(ocr, name)
    found: List[Tuple[int, str, int, str, str, str, str]] = []

    for pat, kind in PLACE_REGEXES:
        for m in pat.finditer(focus):
            raw_place = normalize_ws(m.group(1))
            evidence = sentence_around(focus, m.start(), max_words=35)
            _maybe_add_place(found, m.start(), raw_place, kind, evidence, doc_year, anchor_date, default_page_date, header_loc)

    for m in re.finditer(r"\bborn\s+at\s+([A-Za-z][A-Za-z'’\- ]{1,40}),\s*([A-Za-z][A-Za-z'’\- ]{1,40})", focus, flags=re.I):
        ev = sentence_around(focus, m.start(), max_words=35)
        _maybe_add_place(found, m.start(), m.group(1), "route", ev, doc_year, anchor_date, default_page_date, header_loc)
        _maybe_add_place(found, m.start() + 1, m.group(2), "assoc", ev, doc_year, anchor_date, default_page_date, header_loc)

    for m in re.finditer(r"\bborn\s+at\s+([A-Za-z][A-Za-z'’\- ]{1,40})\s+in\s+([A-Za-z][A-Za-z'’\- ]{1,40})", focus, flags=re.I):
        ev = sentence_around(focus, m.start(), max_words=35)
        _maybe_add_place(found, m.start(), m.group(1), "route", ev, doc_year, anchor_date, default_page_date, header_loc)
        _maybe_add_place(found, m.start() + 1, m.group(2), "assoc", ev, doc_year, anchor_date, default_page_date, header_loc)

    for m in re.finditer(r"\b(?:mi\w*grated|moved)\s+from\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})\s+to\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})", focus, flags=re.I):
        ev = sentence_around(focus, m.start(), max_words=35)
        _maybe_add_place(found, m.start(), m.group(1), "route", ev, doc_year, anchor_date, default_page_date, header_loc)
        _maybe_add_place(found, m.start() + 1, m.group(2), "route", ev, doc_year, anchor_date, default_page_date, header_loc)

    for pat in [
        r"\boriginally\s+lived\s+at\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})",
        r"\blived\s+at\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})",
        r"\bmoved\s+to\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})",
        r"\bre-joined\s+you\s+in\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})",
        r"\b(?:we|i|he|she)\s+reached\s+([A-Za-z][A-Za-z'’\-/ ]{2,60})(?:\s+safely)?",
        r"\blying\s+off\s+([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\barrived\s+at\s+(?:the\s+)?([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\bescaped\s+and\s+arrived\s+at\s+(?:the\s+)?([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\btook\s+refuge\s+at\s+(?:the\s+)?([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\bshipped\s+from\s+([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\bnative\s+of\s+([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\b(?:first|second|present)\s+owner[^\n\.;:]{0,80}?\bof\s+([A-Za-z][A-Za-z'’\-/ ]{2,80})",
        r"\bsold\s+first[^\n\.;:]{0,80}?\bof\s+([A-Za-z][A-Za-z'’\-/ ]{2,80})",
    ]:
        for m in re.finditer(pat, focus, flags=re.I):
            ev = sentence_around(focus, m.start(), max_words=35)
            _maybe_add_place(found, m.start(), m.group(1), "route", ev, doc_year, anchor_date, default_page_date, header_loc)

    if header_loc:
        for m in re.finditer(r"\b(?:took\s+refuge|taken\s+refuge|came\s+to\s+the\s+Agency|recorded\s+at\s+the\s+Political\s+Agency|forward\s+herewith\s+the\s+statement|arrived\s+at\s+the\s+Residency\s+Agency)\b", focus, flags=re.I):
            ev = sentence_around(focus, m.start(), max_words=35)
            _maybe_add_place(found, len(focus) + m.start(), header_loc, "route", ev, doc_year, anchor_date, default_page_date, header_loc)

    found.sort(key=lambda x: x[0])
    merged: Dict[str, Dict[str, Any]] = {}
    route_counter = 0
    for pos, place, base_order, dt_val, dt_conf, time_text, evidence_raw in found:
        evidence = " ".join(evidence_raw.split()[:35])
        key = place.lower()
        cur = merged.get(key)
        if cur is None:
            route_counter += 1 if base_order > 0 else 0
            merged[key] = {
                "place": place,
                "order": route_counter if base_order > 0 else 0,
                "arrival_date": dt_val,
                "date_confidence": dt_conf if dt_val else "",
                "time_text": time_text,
                "evidence": evidence,
                "_pos": pos,
            }
        else:
            if cur.get("order", 0) == 0 and base_order > 0:
                route_counter += 1
                cur["order"] = route_counter
            if not cur.get("arrival_date") and dt_val:
                cur["arrival_date"] = dt_val
                cur["date_confidence"] = dt_conf if dt_val else ""
            if time_text and (not cur.get("time_text") or len(time_text) > len(cur.get("time_text", ""))):
                cur["time_text"] = time_text
            if len(evidence) > len(cur.get("evidence", "")):
                cur["evidence"] = evidence
            cur["_pos"] = min(cur.get("_pos", pos), pos)

    out = sorted(merged.values(), key=lambda x: (0 if x.get("order", 0) > 0 else 1, x.get("order", 0) or 999, x.get("_pos", 0)))
    for row in out:
        if not row.get("arrival_date"):
            row["date_confidence"] = ""
        row.pop("_pos", None)
    return out


def extract_name_scoped_context(ocr: str, name: str) -> str:
    text = ocr or ""
    target = normalize_name(name)
    if not text or not target:
        return text

    aliases = [a for a in _name_aliases(target) if len(a) >= 3]
    sentences = sentence_split(text)
    if sentences:
        selected: List[str] = []
        seen = set()

        def has_alias(sent: str) -> bool:
            low_sent = _strip_accents(sent.lower())
            return any(re.search(rf"\b{re.escape(_strip_accents(a.lower()))}\b", low_sent) for a in aliases)

        for i, sent in enumerate(sentences):
            if not has_alias(sent):
                continue
            idxs = []
            if i - 1 >= 0 and re.search(r"\b(?:the\s+following\s+refugee\s+slaves|first\s+batch|second\s+batch|persons\s+concerned)\b", sent, flags=re.I):
                idxs.append(i - 1)
            idxs.append(i)
            if i + 1 < len(sentences) and re.match(r"^(?:i|he|she|they|from there|thence|after|then|later|subsequently|was|were)\b", sentences[i + 1].lower()):
                idxs.append(i + 1)
            for j in idxs:
                ss = normalize_ws(sentences[j])
                if ss and ss not in seen:
                    selected.append(ss)
                    seen.add(ss)

        if selected:
            return "\n\n".join(selected)[:12000]

    focus = normalize_ws(extract_focus_text_for_name(text, target))
    return focus[:12000] if focus else text


def deterministic_meta(ocr: str, name: str, page: int, report_type: str) -> Dict[str, Any]:
    scoped = extract_name_scoped_context(ocr, name)
    low = (scoped or "").lower()
    crime_type = ""
    if re.search(r"\b(kidnapped|abducted)\b", low):
        crime_type = "kidnapping"
    elif re.search(r"\btraffick", low):
        crime_type = "trafficking"
    elif re.search(r"\billegal detention\b|\bdetained against (?:his|her|their|my) will\b", low):
        crime_type = "illegal detention"
    elif re.search(r"\bsold\b", low):
        crime_type = "sale"

    whether_abuse = ""
    if re.search(r"\b(beat|beaten|abused|ill-treated|ill treated|forced|cruel|violence|assault|maltreat|maltreated|manacles?|imprison|putting me to prison|miseries of .* mistress)\b", low):
        whether_abuse = "yes"
    elif re.search(r"\bno abuse\b|\bnot abused\b", low):
        whether_abuse = "no"

    conflict_type = ""
    if re.search(r"\b(re-enslave|re enslave|manumission certificate|requesting to be manumitted|wished to be manumitted|wishes that .*? manumission certificate|grant .* manumission certificate|free woman|free man|freedom)\b", low):
        conflict_type = "manumission dispute"
    elif re.search(r"\bheirs?\b.*\bcomplain|property of|ownership|claim\b", low):
        conflict_type = "ownership dispute"
    elif re.search(r"\bkidnapped|abducted\b", low):
        conflict_type = "kidnapping case"

    trial = ""
    if re.search(r"\b(?:requesting|wished|wish(?:es)?|asking)\s+to\s+be\s+manumitted\b|\brequesting .* manumission certificate\b|\bgrant .* manumission certificate\b", low):
        trial = "manumission certificate requested" if "certificate" in low else "manumission requested"
    elif re.search(r"\b(?:ascertained|found|proved|established)\b[^.]{0,80}\b(?:free man|free woman|free person)\b", low):
        trial = "free man confirmed"
    elif re.search(r"\bwas\s+given\s+a\s+manumission\s+certificate\b|\bwere\s+given\s+manumission\s+certificates\b", low):
        trial = "manumission granted"
    elif re.search(r"\breleased\b", low):
        trial = "released"

    amount_paid = find_amount_paid(scoped) or ""

    return {
        "Name": name,
        "Page": page,
        "Report Type": report_type,
        "Crime Type": crime_type,
        "Whether abuse": whether_abuse,
        "Conflict Type": conflict_type,
        "Trial": trial,
        "Amount paid": amount_paid,
    }


def merge_named_candidates(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for group in groups:
        for item in group or []:
            name = normalize_name(item.get("name", ""))
            if not is_likely_personal_name(name):
                continue
            evidence = normalize_ws(item.get("evidence", ""))
            matched = False
            for prev in out:
                if names_maybe_same_person(prev["name"], name):
                    prev["name"] = choose_preferred_name(prev["name"], name)
                    if len(evidence) > len(prev.get("evidence", "")):
                        prev["evidence"] = evidence
                    matched = True
                    break
            if not matched:
                out.append({"name": name, "evidence": evidence})
    return out


def merge_places(model_places: List[Dict[str, Any]], rule_places: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for src in [rule_places or [], model_places or []]:
        for p in src:
            place = normalize_place(p.get("place", ""))
            if not is_valid_place(place):
                continue
            key = place.lower()
            try:
                order = int(p.get("order", 0))
            except Exception:
                order = 0
            candidate = {
                "place": place,
                "order": order,
                "arrival_date": normalize_ws(str(p.get("arrival_date") or "")),
                "date_confidence": normalize_ws(str(p.get("date_confidence") or "")),
                "time_text": normalize_ws(str(p.get("time_text") or "")),
                "evidence": normalize_ws(str(p.get("evidence") or "")),
            }
            if not candidate["arrival_date"]:
                candidate["date_confidence"] = ""
            cur = merged.get(key)
            if cur is None:
                merged[key] = candidate
            else:
                if cur.get("order", 0) == 0 and order > 0:
                    cur["order"] = order
                elif order > 0 and cur.get("order", 0) > 0:
                    cur["order"] = min(cur["order"], order)
                if not cur.get("arrival_date") and candidate.get("arrival_date"):
                    cur["arrival_date"] = candidate["arrival_date"]
                    cur["date_confidence"] = candidate.get("date_confidence", "")
                if candidate.get("time_text") and (not cur.get("time_text") or len(candidate.get("time_text", "")) > len(cur.get("time_text", ""))):
                    cur["time_text"] = candidate.get("time_text", "")
                if len(candidate.get("evidence", "")) > len(cur.get("evidence", "")):
                    cur["evidence"] = candidate.get("evidence", "")
    positives = sorted([p for p in merged.values() if p.get("order", 0) > 0], key=lambda x: x["order"])
    for i, p in enumerate(positives, start=1):
        p["order"] = i
    zeroes = [p for p in merged.values() if p.get("order", 0) == 0]
    return positives + sorted(zeroes, key=lambda x: x["place"])


def dedupe_page_rows(detail_rows: List[Dict[str, Any]], place_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    groups: List[Dict[str, Any]] = []
    for row in detail_rows:
        nm = normalize_name(str(row.get("Name") or ""))
        if not is_likely_personal_name(nm):
            continue
        placed = False
        for grp in groups:
            if names_maybe_same_person(grp["name"], nm):
                grp["name"] = choose_preferred_name(grp["name"], nm)
                grp["rows"].append(row)
                placed = True
                break
        if not placed:
            groups.append({"name": nm, "rows": [row]})

    name_map: Dict[str, str] = {}
    dedup_detail: List[Dict[str, Any]] = []
    for grp in groups:
        best_name = grp["name"]
        rows = grp["rows"]
        merged = dict(rows[0])
        merged["Name"] = best_name
        for r in rows[1:]:
            for col in DETAIL_COLUMNS:
                if col == "Name":
                    continue
                if not merged.get(col) and r.get(col):
                    merged[col] = r.get(col)
                elif col == "Whether abuse" and r.get(col) == "yes":
                    merged[col] = "yes"
                elif col == "Trial":
                    existing = str(merged.get(col) or "").lower()
                    incoming = str(r.get(col) or "")
                    if not existing and incoming:
                        merged[col] = incoming
                    elif existing == "manumission requested" and incoming == "manumission certificate requested":
                        merged[col] = incoming
        dedup_detail.append(merged)
        for r in rows:
            name_map[normalize_name(str(r.get("Name") or ""))] = best_name

    grouped_places: Dict[Tuple[str, Any], List[Dict[str, Any]]] = defaultdict(list)
    for r in place_rows:
        nm = normalize_name(str(r.get("Name") or ""))
        if nm in name_map:
            newr = dict(r)
            newr["Name"] = name_map[nm]
            grouped_places[(newr["Name"], newr.get("Page"))].append(newr)

    dedup_places: List[Dict[str, Any]] = []
    for (_, _), rows in grouped_places.items():
        merged_places = merge_places(
            [{
                "place": r.get("Place", ""),
                "order": r.get("Order", 0),
                "arrival_date": r.get("Arrival Date", ""),
                "date_confidence": r.get("Date Confidence", ""),
                "time_text": r.get("Time Info", ""),
                "evidence": "",
            } for r in rows if r.get("Place")],
            []
        )
        if not merged_places:
            dedup_places.append(blank_place_row(rows[0].get("Name", ""), int(rows[0].get("Page") or 0)))
        else:
            for p in merged_places:
                dedup_places.append({
                    "Name": rows[0].get("Name", ""),
                    "Page": rows[0].get("Page", ""),
                    "Place": p.get("place", ""),
                    "Order": p.get("order", ""),
                    "Arrival Date": p.get("arrival_date", ""),
                    "Date Confidence": p.get("date_confidence", "") if p.get("arrival_date") else "",
                    "Time Info": p.get("time_text", ""),
                })
    return dedup_detail, dedup_places

def fill_meta_from_model(base: Dict[str, Any], model_meta: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    mapping = {
        "crime_type": "Crime Type",
        "whether_abuse": "Whether abuse",
        "conflict_type": "Conflict Type",
        "trial": "Trial",
        "amount_paid": "Amount paid",
    }
    for mk, ck in mapping.items():
        val = model_meta.get(mk)
        if val is None:
            continue
        sval = normalize_ws(str(val))
        if not sval or sval.lower() in {"null", "none", "unknown", "n/a"}:
            continue
        if ck == "Whether abuse":
            sval = sval.lower()
            if sval not in {"yes", "no", ""}:
                continue
            if sval == "yes" or not out.get(ck):
                out[ck] = sval
            continue
        out[ck] = sval
    return out


# ------------------- CONTEXT ENRICHMENT -------------------
def parse_combined_ocr_file(path: str) -> Dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    txt = pathlib.Path(path).read_text(encoding="utf-8", errors="ignore")
    out: Dict[int, str] = {}
    matches = list(re.finditer(r"^Page:\s*(\d+)\.txt\s*$", txt, flags=re.M))
    for i, m in enumerate(matches):
        page = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        chunk = txt[start:end]
        chunk = re.sub(r"View on the Qatar Digital Library:.*", "", chunk)
        chunk = chunk.replace("------------------------", "\n")
        chunk = normalize_ws(chunk)
        if chunk:
            out[page] = chunk
    return out



def looks_like_new_case(text: str) -> bool:
    head = normalize_ws((text or "")[:220])
    if not head:
        return False
    return bool(re.search(r"\bSubject\s+\d+\b|\bManumission of\b|\bCorrespondence\b|\bTelegram Code\b|\bNo\.\s*\d+\s+of\s+\d{4}\b", head, flags=re.I))


def _case_signature_tokens(text: str) -> List[str]:
    head = normalize_ws((text or "")[:700]).lower()
    head = head.split("first batch of slaves")[0]
    toks = [t for t in re.findall(r"[a-z]{4,}", head) if t not in {
        "from", "date", "submitted", "immediately", "anchored", "board", "slave", "slaves", "statement",
        "residency", "agent", "political", "resident", "letter", "dated", "page", "this", "that", "with",
        "have", "when", "they", "them", "their", "there", "were", "was", "been", "which", "would",
    }]
    return toks[:25]


def same_case_neighbor(a: str, b: str) -> bool:
    ta = set(_case_signature_tokens(a))
    tb = set(_case_signature_tokens(b))
    if not ta or not tb:
        return False
    overlap = len(ta & tb)
    return overlap >= 4 or ("khassab" in ta and "khassab" in tb and "fugitive" in ta and "fugitive" in tb)


def _name_aliases(name: str) -> List[str]:
    n = normalize_name(name).lower()
    toks = [t for t in n.split() if t]
    aliases = {n}
    if len(toks) >= 2:
        aliases.add(" ".join(toks[:2]))
        aliases.add(toks[0] + " " + toks[-1])
    if toks:
        aliases.add(toks[0])
    return sorted(a for a in aliases if a)


def collect_case_context(page: int, current_ocr: str, name: str, ordered_pages: List[int], page_texts: Dict[int, str]) -> str:
    if not page_texts or page not in ordered_pages:
        return current_ocr
    idx = ordered_pages.index(page)
    chosen = []
    aliases = _name_aliases(name)
    current_type = classify_document_page(current_ocr)
    current_quality, _ = score_ocr_quality(current_ocr)

    def alias_hit(txt: str) -> bool:
        low = _strip_accents((txt or "").lower())
        return any(a and re.search(rf"\b{re.escape(_strip_accents(a.lower()))}\b", low) for a in aliases if len(a) >= 3)

    if idx - 1 >= 0:
        prev = page_texts.get(ordered_pages[idx - 1], "")
        if prev and (alias_hit(prev) or same_case_neighbor(prev, current_ocr)):
            chosen.append(prev)

    chosen.append(page_texts.get(page, current_ocr))

    if idx + 1 < len(ordered_pages):
        nxt = page_texts.get(ordered_pages[idx + 1], "")
        if nxt:
            nxt_type = classify_document_page(nxt)
            nxt_quality, _ = score_ocr_quality(nxt)
            continuation_like = re.match(r"^(to the fact|with my mother|she |he |at the end|during |when |i )", normalize_ws(nxt).lower()) is not None
            include_next = False
            if alias_hit(nxt) or same_case_neighbor(current_ocr, nxt):
                include_next = True
            elif current_type == "narrative_statement" and nxt_type in {"narrative_statement", "unknown"} and nxt_quality != "garbled" and continuation_like and current_quality != "garbled":
                include_next = True
            if include_next:
                chosen.append(nxt)

    for j in range(idx + 2, min(idx + 5, len(ordered_pages))):
        txt2 = page_texts.get(ordered_pages[j], "")
        if not txt2:
            continue
        txt_quality, _ = score_ocr_quality(txt2)
        if txt_quality == "garbled":
            break
        if alias_hit(txt2) or same_case_neighbor(current_ocr, txt2):
            chosen.append(txt2)
        elif looks_like_new_case(txt2):
            break
    return "\n\n".join(dict.fromkeys(chosen))

# ------------------- MODEL PASS HELPERS -------------------

def model_named_slaves(ocr: str, stats: Dict[str, int]) -> List[Dict[str, str]]:
    schema = '{"named_slaves": [{"name": "...", "evidence": "..."}]}'
    rows_1: List[Dict[str, str]] = []
    rows_2: List[Dict[str, str]] = []

    obj = call_json_prompt(NAME_PASS_PROMPT.format(ocr=ocr), stats, schema, num_predict=700)
    if isinstance(obj, dict):
        for item in obj.get("named_slaves") or []:
            name = normalize_name(str(item.get("name") or ""))
            if not is_likely_personal_name(name):
                continue
            rows_1.append({"name": name, "evidence": normalize_ws(str(item.get("evidence") or ""))})

    obj = call_json_prompt(NAME_RECALL_PROMPT.format(ocr=ocr), stats, schema, num_predict=700)
    if isinstance(obj, dict):
        for item in obj.get("named_slaves") or []:
            name = normalize_name(str(item.get("name") or ""))
            if not is_likely_personal_name(name):
                continue
            rows_2.append({"name": name, "evidence": normalize_ws(str(item.get("evidence") or ""))})

    return merge_named_candidates(rows_1, rows_2)


def _parse_places_obj(obj: Any, doc_year: Optional[int], anchor_date: Optional[dt.date] = None, header_loc: str = "") -> List[Dict[str, Any]]:
    places: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        for p in obj.get("places") or []:
            place = normalize_place(str(p.get("place") or ""))
            if not is_valid_place(place):
                continue
            evidence = normalize_ws(str(p.get("evidence") or ""))
            if evidence_disqualifies_place(evidence, place, header_loc):
                continue
            try:
                order = int(p.get("order", 0))
            except Exception:
                order = 0
            arrival_date = p.get("arrival_date")
            arrival_date = "" if arrival_date in (None, "null") else normalize_ws(str(arrival_date))
            inferred_conf = ""
            if arrival_date:
                parsed_date, inferred_conf = to_iso_date(arrival_date, doc_year)
                arrival_date = parsed_date or arrival_date
            if not arrival_date:
                arrival_date, inferred_conf = extract_nearby_date(evidence, doc_year, anchor_date)
            date_confidence = normalize_ws(str(p.get("date_confidence") or "")) or inferred_conf
            if not arrival_date:
                date_confidence = ""
            time_text = normalize_ws(str(p.get("time_text") or "")) or extract_time_text(evidence)
            places.append({
                "place": place,
                "order": order,
                "arrival_date": arrival_date,
                "date_confidence": date_confidence,
                "time_text": time_text,
                "evidence": evidence,
            })
    return places


def model_places_for_name(ocr: str, name: str, stats: Dict[str, int]) -> List[Dict[str, Any]]:
    doc_year = extract_doc_year(ocr)
    anchor_date = extract_anchor_date(ocr, doc_year)
    header_loc = extract_header_location(ocr)
    schema = '{"name":"%s","places":[{"place":"...","order":1,"arrival_date":null,"date_confidence":"unknown","time_text":null,"evidence":"..."}]}' % name
    obj = call_json_prompt(PLACE_PASS_PROMPT.format(name=name, ocr=ocr), stats, schema, num_predict=1100)
    places = _parse_places_obj(obj, doc_year, anchor_date, header_loc)
    if len(places) <= 1:
        obj2 = call_json_prompt(PLACE_RECALL_PROMPT.format(name=name, ocr=ocr), stats, schema, num_predict=1100)
        places = merge_places(_parse_places_obj(obj2, doc_year, anchor_date, header_loc), places)
    for p in places:
        if not p.get("arrival_date"):
            p["date_confidence"] = ""
    return places


def model_meta_for_name(ocr: str, name: str, page: int, report_type: str, stats: Dict[str, int]) -> Dict[str, Any]:
    schema = '{"name":"%s","page":%s,"report_type":"%s","crime_type":null,"whether_abuse":"","conflict_type":null,"trial":null,"amount_paid":null,"evidence":null}' % (name, page, report_type)
    obj = call_json_prompt(META_PASS_PROMPT.format(name=name, page=page, report_type=report_type, ocr=ocr), stats, schema, num_predict=700)
    if isinstance(obj, dict):
        return obj
    return {}

# ------------------- PAGE PROCESSING -------------------

def process_page(current_ocr: str, page: int, filename: str, report_type: str, stats: Dict[str, int], logger: logging.Logger,
                 ordered_pages: Optional[List[int]] = None, page_texts: Optional[Dict[int, str]] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
    page_type = classify_document_page(current_ocr)
    if page_type in {"index", "record_metadata"}:
        return [], [], "skip_index_page"

    ocr_quality, quality_info = score_ocr_quality(current_ocr)
    if ocr_quality == "garbled" and not quality_info.get("strong_anchor"):
        return [], [], "skip_bad_ocr"

    doc_year = extract_doc_year(current_ocr)
    low = (current_ocr or "").lower()
    rule_names = deterministic_named_slaves(current_ocr)

    if not rule_names and page_type == "administrative_memo":
        shortish = len(normalize_ws(current_ocr)) <= 900
        explicit_name_anchor = bool(re.search(r"\b(?:named\s+[A-Z]|namely|slave\s*no\.?\s*\d+|statement\s+made\s+by|statement\s+of(?:\s+slave)?|grant\s+.+?manumission\s+certificate)\b", current_ocr, flags=re.I))
        if shortish and not explicit_name_anchor:
            return [], [], "skip_no_named_slave"

    model_names = []
    try:
        model_names = model_named_slaves(current_ocr, stats)
    except Exception as e:
        logger.warning("Page %s name-pass failed: %s", page, e)

    names = merge_named_candidates(model_names, rule_names)

    if not names:
        return [], [], "skip_no_named_slave"

    detail_rows: List[Dict[str, Any]] = []
    place_rows: List[Dict[str, Any]] = []

    for item in names:
        name = item["name"]
        if not is_likely_personal_name(name):
            continue

        case_ocr = collect_case_context(page, current_ocr, name, ordered_pages or [], page_texts or {})
        case_doc_year = extract_doc_year(case_ocr) or doc_year
        name_context = extract_name_scoped_context(case_ocr, name) or case_ocr

        base_meta = deterministic_meta(name_context, name, page, report_type)
        model_meta = {}
        try:
            model_meta = model_meta_for_name(name_context, name, page, report_type, stats)
        except Exception as e:
            logger.warning("Page %s meta-pass failed for %s: %s", page, name, e)
        meta = fill_meta_from_model(base_meta, model_meta)

        canonical_name = canonicalize_name_against_context(str(meta.get("Name") or name), case_ocr)
        meta["Name"] = canonical_name

        if meta.get("Trial", "").lower() == "free man confirmed" and re.search(r"\brequest(?:ing)?\s+to\s+be\s+manumitted\b|\bwished\s+to\s+be\s+manumitted\b", name_context, flags=re.I):
            meta["Trial"] = "manumission requested"
        if not meta.get("Conflict Type") and re.search(r"\brequest(?:ing)?\s+to\s+be\s+manumitted\b|\bwished\s+to\s+be\s+manumitted\b|\bmanumission certificate\b", name_context, flags=re.I):
            meta["Conflict Type"] = "manumission dispute"
        if not meta.get("Whether abuse") and re.search(r"\b(miseries of .* mistress|maltreat|beating|putting me to prison|manacles?)\b", name_context, flags=re.I):
            meta["Whether abuse"] = "yes"

        detail_rows.append(meta)

        allow_route = page_supports_personal_route(page_type, case_ocr, ocr_quality)
        det_places = deterministic_places_for_page(case_ocr, canonical_name, case_doc_year) if allow_route else []
        model_places = []
        if allow_route:
            try:
                model_places = model_places_for_name(case_ocr, canonical_name, stats)
            except Exception as e:
                logger.warning("Page %s place-pass failed for %s: %s", page, canonical_name, e)
        places = merge_places(det_places, model_places)

        if not places:
            place_rows.append(blank_place_row(canonical_name, page))
        else:
            for p in places:
                place_rows.append({
                    "Name": canonical_name,
                    "Page": page,
                    "Place": p.get("place", ""),
                    "Order": p.get("order", ""),
                    "Arrival Date": p.get("arrival_date", ""),
                    "Date Confidence": p.get("date_confidence", "") if p.get("arrival_date") else "",
                    "Time Info": p.get("time_text", ""),
                })

    detail_rows = [r for r in detail_rows if is_likely_personal_name(str(r.get("Name") or ""))]
    place_rows = [r for r in place_rows if is_likely_personal_name(str(r.get("Name") or ""))]

    detail_rows, place_rows = dedupe_page_rows(detail_rows, place_rows)

    valid_names = {normalize_name(r["Name"]) for r in detail_rows}
    place_rows = [r for r in place_rows if normalize_name(r.get("Name", "")) in valid_names]

    seen_place_names = {normalize_name(r.get("Name", "")) for r in place_rows}
    for r in detail_rows:
        nm = normalize_name(r["Name"])
        if nm not in seen_place_names:
            place_rows.append(blank_place_row(r["Name"], int(r.get("Page", page) or page)))
            seen_place_names.add(nm)

    if not detail_rows:
        return [], [], "skip_no_named_slave"

    return detail_rows, place_rows, "ok"


def assert_runtime_integrity() -> None:
    required = [
        "normalize_name",
        "is_valid_place",
        "merge_places",
        "dedupe_page_rows",
        "process_page",
        "deterministic_meta",
        "fill_meta_from_model",
    ]
    missing = [name for name in required if not callable(globals().get(name))]
    if missing:
        raise RuntimeError(f"Missing required helpers: {', '.join(missing)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Multipass slave extraction with incremental CSV output.")
    parser.add_argument("--in_dir", default="/data/input")
    parser.add_argument("--out_dir", default="/data/output")
    parser.add_argument("--text_out_dir", default="/data/text_out")  # compatibility only
    parser.add_argument("--log_dir", default="/data/logs")
    parser.add_argument("--report_type", default=None, help="Optional fixed report type; if omitted, infer from first page.")
    parser.add_argument("--save_debug_json", action="store_true")
    parser.add_argument("--combined_ocr_path", default=None, help="Optional combined OCR file with Page: XXXX.txt sections.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    logger = _setup_logger(args.log_dir)
    assert_runtime_integrity()

    detail_path = os.path.join(args.out_dir, "Detailed info.csv")
    place_path = os.path.join(args.out_dir, "name place.csv")
    status_path = os.path.join(args.log_dir, "run_status.csv")
    debug_dir = os.path.join(args.out_dir, "debug_json")
    if args.save_debug_json:
        os.makedirs(debug_dir, exist_ok=True)

    files = sorted([p for p in pathlib.Path(args.in_dir).glob("*.txt") if re.fullmatch(r"\d+", p.stem)])
    if not files:
        logger.error("No .txt files found in %s", args.in_dir)
        write_csv(detail_path, [], DETAIL_COLUMNS)
        write_csv(place_path, [], PLACE_COLUMNS)
        write_csv(status_path, [], STATUS_COLUMNS)
        return

    first_page_text = "\n\n".join(fp.read_text(encoding="utf-8", errors="ignore") for fp in files[: min(5, len(files))])
    report_type = args.report_type or infer_report_type_from_text(first_page_text)
    logger.info("Using report type: %s", report_type)

    combined_candidate = args.combined_ocr_path
    if not combined_candidate:
        auto = pathlib.Path(args.in_dir) / "PDF extracted text.txt"
        if auto.exists():
            combined_candidate = str(auto)
    combined_map = parse_combined_ocr_file(combined_candidate) if combined_candidate else {}
    ordered_pages: List[int] = []
    page_texts: Dict[int, str] = {}
    for fp in files:
        pnum = int(re.sub(r"\D", "", fp.stem) or 0)
        ordered_pages.append(pnum)
        page_texts[pnum] = combined_map.get(pnum) or fp.read_text(encoding="utf-8", errors="ignore")

    all_detail_rows: List[Dict[str, Any]] = []
    all_place_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []
    state = {
        "started_at": dt.datetime.utcnow().isoformat() + "Z",
        "report_type": report_type,
        "processed_pages": [],
    }

    # Initialize files immediately
    write_csv(detail_path, all_detail_rows, DETAIL_COLUMNS)
    write_csv(place_path, all_place_rows, PLACE_COLUMNS)
    write_csv(status_path, status_rows, STATUS_COLUMNS)
    write_state(args.log_dir, state)

    for fp in files:
        t0 = time.time()
        filename = fp.name
        page_str = re.sub(r"\D", "", fp.stem) or fp.stem
        try:
            page = int(page_str)
        except Exception:
            page = 0
        ocr = fp.read_text(encoding="utf-8", errors="ignore")
        stats = {"model_calls": 0, "extract_calls": 0, "repair_calls": 0}
        note = ""
        status = "ok"
        detail_rows: List[Dict[str, Any]] = []
        place_rows: List[Dict[str, Any]] = []

        try:
            detail_rows, place_rows, status = process_page(ocr, page, filename, report_type, stats, logger, ordered_pages, page_texts)
            if status == "ok":
                all_detail_rows.extend(detail_rows)
                all_place_rows.extend(place_rows)
            else:
                note = status

            if args.save_debug_json:
                dbg = {
                    "page": page,
                    "filename": filename,
                    "report_type": report_type,
                    "detail_rows": detail_rows,
                    "place_rows": place_rows,
                    "status": status,
                    "stats": stats,
                }
                with open(os.path.join(debug_dir, f"{fp.stem}.json"), "w", encoding="utf-8") as f:
                    json.dump(dbg, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.exception("Page %s failed", page)
            status = "error"
            note = str(e)

        # Incremental save after EVERY page
        write_csv(detail_path, all_detail_rows, DETAIL_COLUMNS)
        write_csv(place_path, all_place_rows, PLACE_COLUMNS)

        elapsed = round(time.time() - t0, 2)
        status_rows.append({
            "page": page,
            "filename": filename,
            "status": status,
            "named_slaves": len({r['Name'] for r in detail_rows}) if detail_rows else 0,
            "detail_rows": len(detail_rows),
            "place_rows": len(place_rows),
            "model_calls": stats["model_calls"],
            "extract_calls": stats["extract_calls"],
            "repair_calls": stats["repair_calls"],
            "elapsed_seconds": elapsed,
            "note": note,
        })
        write_csv(status_path, status_rows, STATUS_COLUMNS)

        state["processed_pages"].append({
            "page": page,
            "filename": filename,
            "status": status,
            "named_slaves": len({r['Name'] for r in detail_rows}) if detail_rows else 0,
            "model_calls": stats["model_calls"],
            "elapsed_seconds": elapsed,
        })
        write_state(args.log_dir, state)
        logger.info("Processed page=%s status=%s named=%s model_calls=%s elapsed_seconds=%s", page, status,
                    len({r['Name'] for r in detail_rows}) if detail_rows else 0, stats["model_calls"], elapsed)
        print(f"[page {page}] status={status} named={len({r['Name'] for r in detail_rows}) if detail_rows else 0} "
              f"model_calls={stats['model_calls']} elapsed_seconds={elapsed}", flush=True)

    state["finished_at"] = dt.datetime.utcnow().isoformat() + "Z"
    write_state(args.log_dir, state)
    logger.info("Done. Detailed rows=%s Place rows=%s", len(all_detail_rows), len(all_place_rows))


if __name__ == "__main__":
    main()
