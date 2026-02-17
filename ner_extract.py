#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_extract_updated_v6.py

Extract enslaved persons and *explicit, evidenced* place timeline events from OCR documents
using a local Ollama model, then post-process into a clean chronological place sequence.

This version focuses on COMPLETENESS without sacrificing correctness:
- The model outputs *events* (presence/movement/stay/etc.). We never force ∅ origins/destinations.
- Strict validation is applied AFTER deterministic cleaning (so schema-drift like place=null is dropped, not fatal).
- A rule-based narrator augmentation pass adds clearly stated places for first-person "statement made by ..." narratives
  when the model under-extracts (common with local models). This is conservative: it only uses explicit place-linking phrases.

Run:
  python batch_extract_updated_v6.py --in_dir ocr_text --out_dir ocr_text_out --text_out_dir ocr_text_out_text
"""

import os, json, re, time, argparse, difflib, logging, csv, datetime, pathlib
from typing import Optional
import requests
from requests.exceptions import ReadTimeout, ConnectionError

# ------------------- OLLAMA CONFIG -------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")  # change if needed

# Timeouts:
REQUEST_TIMEOUT = (10, 600)  # connect, read
MAX_CALL_RETRIES = 3
RETRY_BACKOFF_SECONDS = 15  # 15s, 30s, 45s ...

# ------------------- RUNTIME / LOGGING -------------------
# Logs are written to the *current working directory* by default ("root" of your run).
DEFAULT_LOG_DIR = os.environ.get("RUN_LOG_DIR", ".")
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1200"))
DEFAULT_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX")  # optional
DEFAULT_NUM_CTX = int(DEFAULT_NUM_CTX) if DEFAULT_NUM_CTX and DEFAULT_NUM_CTX.isdigit() else None

# Reuse one HTTP session (faster than creating a new TCP connection per request)
_OLLAMA_SESSION = requests.Session()
_OLLAMA_SESSION.headers.update({"Connection": "keep-alive"})

def _setup_logger(log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("batch_extract")
    if logger.handlers:
        return logger  # already configured

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    # file log
    fh = logging.FileHandler(os.path.join(log_dir, "run.log"), encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    # console log
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def _write_state(log_dir: str, state: dict):
    tmp = os.path.join(log_dir, "run_state.json.tmp")
    path = os.path.join(log_dir, "run_state.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

_INDEX_LINE_PAT = re.compile(r"^\s*(?:\(?\d{1,4}\)?[\).]|[A-Z]\.|[ivxlcdm]{1,6}[\).])\s+", re.I)

def looks_like_index_page(ocr: str) -> bool:
    """Heuristic: lots of short, enumerated lines => index/contents page."""
    if not ocr:
        return False
    lines = [ln.strip() for ln in ocr.splitlines() if ln.strip()]
    if len(lines) < 40:
        return False
    sample = lines[:160]
    enum = sum(1 for ln in sample if _INDEX_LINE_PAT.match(ln))
    short = sum(1 for ln in sample if len(ln) <= 90)
    # also treat as index if it literally says "index"/"contents" and is liney
    low = ocr.lower()
    has_header = ("contents" in low) or ("index" in low)
    return (enum >= 12 and (short / max(1, len(sample))) >= 0.60) or (has_header and (short / max(1, len(sample))) >= 0.60)

def chunk_text_by_lines(text: str, lines_per_chunk: int = 70):
    lines = text.splitlines()
    chunks = []
    buf = []
    for ln in lines:
        buf.append(ln)
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

def merge_partial_objects(objs: list, doc_id: str) -> dict:
    """Merge multiple per-chunk extraction JSONs into a single doc-level JSON."""
    out = {"doc_id": doc_id, "document_date": None, "people": []}
    by_key = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        if out["document_date"] is None and o.get("document_date"):
            out["document_date"] = o.get("document_date")
        for p in (o.get("people") or []):
            if not isinstance(p, dict):
                continue
            name = (p.get("name") or "").strip()
            if not name:
                continue
            key = normalize_name(name).lower()
            if key not in by_key:
                by_key[key] = {
                    "name": name,
                    "enslaved_status": p.get("enslaved_status") or "strong_inferred",
                    "enslaved_evidence": p.get("enslaved_evidence") or "",
                    "events": list(p.get("events") or []),
                }
            else:
                # prefer explicit status
                if (p.get("enslaved_status") or "") == "explicit":
                    by_key[key]["enslaved_status"] = "explicit"
                # keep the longer evidence snippet if present
                ev = (p.get("enslaved_evidence") or "").strip()
                if len(ev) > len(by_key[key].get("enslaved_evidence") or ""):
                    by_key[key]["enslaved_evidence"] = ev
                by_key[key]["events"].extend(list(p.get("events") or []))
    out["people"] = list(by_key.values())
    return out

# ------------------- PLACE NORMALIZATION -------------------
# Add/edit mappings as you discover systematic OCR variants.
PLACE_MAP = {
    # UAE
    "shargah": "Sharjah",
    "sharjah": "Sharjah",
    "abu dhabi": "Abu Dhabi",
    "ras ul khaimah": "Ras al Khaimah",
    "ras ul khaima": "Ras al Khaimah",
    "ras al khaimah": "Ras al Khaimah",
    "rasul khaimah": "Ras al Khaimah",
    "ras ulkhaimah": "Ras al Khaimah",
    "umm al quwain": "Umm al Quwain",
    "umm ul quwain": "Umm al Quwain",
    "umm el quwain": "Umm al Quwain",
    "jumairah": "Jumeirah",
    "jumeirah": "Jumeirah",
    "shamdaghah": "Shamdaghah",   # keep as label; optionally map to "Shindagha" if you prefer
    "shindagha": "Shindagha",

    # Iran
    "bushire": "Bushehr",
    "bushehr": "Bushehr",
    "busheir": "Bushehr",

    # Common narrative labels
    "oman coast": "Oman Coast",
    "batinah district town": "Batinah district (town unknown)",
    "batinah district (town unknown)": "Batinah district (town unknown)",
    "bishakird": "Bishakird",
}

NON_GEO_PLACEHOLDERS = {
    "∅", "unknown", "unclear", "not known", "n/a", "na", "nil", "none",
    "agency", "residency", "residency agency", "british residency",
    "political resident", "resident", "residency agent",
    "on board", "onboard",
    # Vague placeholders (not named places)
    "diving", "diving location", "diving place", "diving grounds", "pearl diving", "pearling",
    "at sea",
}


# Reject clause-like place strings (common OCR failure mode where a whole sentence fragment is captured as a "place")
MAX_PLACE_WORDS = 6
BAD_PLACE_TOKEN_PAT = re.compile(
    r"\b(and|to|from|therefore|told|landed|kept|captured|bought|sold|sent|took|arrived|reached|earn|livelihood|godown|days|later|recorded|aged|statement|made)\b",
    re.I
)

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12
}

# ------------------- NAME NORMALIZATION / FILTERING -------------------
ROLE_TITLE_PAT = re.compile(
    r"\b(residency agent|political resident|treasury accounts officer|chief|shaikh|sheikh|major|captain|lieutenant|lt\.?|mr\.?|mrs\.?|miss|nakhuda)\b",
    re.I
)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def normalize_name(raw: str) -> str:
    """
    Conservative normalization:
    - Strip anything after '| Role:' or similar suffixes.
    - Normalize spacing and punctuation.
    - Normalize common Arabic kinship particles to a consistent form.
    """
    if not raw:
        return ""
    s = raw
    s = re.split(r"\|\s*role\s*:", s, flags=re.I)[0]
    s = normalize_ws(s)
    s = s.strip(" ,.;:[]{}\"'")
    # normalize ibn/bin/bint variants
    s = re.sub(r"\b(ibn|bin)\b", "bin", s, flags=re.I)
    s = re.sub(r"\b(bint)\b", "bint", s, flags=re.I)
    s = re.sub(r"\b(abu)\b", "Abu", s, flags=re.I)
    s = re.sub(r"\b(umm)\b", "Umm", s, flags=re.I)
    # Fix common OCR merge: "Hassunraged" -> "Hassun"
    s = re.sub(r"\b([A-Za-z]{2,})r?aged\b.*$", r"\1", s, flags=re.I)
    s = normalize_ws(s)
    return s

def is_likely_personal_name(name: str) -> bool:
    """
    Filter out obvious non-person entries, generic roles, and empty names.
    Intentionally conservative.
    """
    if not name:
        return False
    n = normalize_ws(name)
    if len(n) < 2:
        return False
    # reject if looks like a pure title/role line
    if ROLE_TITLE_PAT.search(n) and len(n.split()) <= 5:
        return False
    # reject generic placeholders
    low = n.lower()
    if low in ("woman", "man", "boy", "girl", "slave", "unknown", "unnamed"):
        return False
    return True

# ------------------- SHIP NAME DETECTION -------------------
SHIP_PAT = re.compile(r"\b(h\.?\s*m\.?\s*s\.?|s\.?\s*s\.?|steamer|dhow)\b", re.I)

def normalize_ship_name(place: str) -> str:
    s = normalize_ws(place)
    s = re.sub(r"\bH\s*\.?\s*M\s*\.?\s*S\s*\.?\b", "H.M.S.", s, flags=re.I)
    s = re.sub(r"\bS\s*\.?\s*S\s*\.?\b", "S.S.", s, flags=re.I)
    return s

def is_ship(place: str) -> bool:
    return bool(SHIP_PAT.search(place or ""))

def normalize_place(raw: str) -> str:
    if not raw:
        return ""
    s = normalize_ws(raw)
    s = s.strip(" ,.;:[]{}\"'")
    if not s:
        return ""
    if is_ship(s):
        return normalize_ship_name(s)

    low = s.lower()
    low = low.replace('-', ' ')

    # Normalize a few common narrative phrases
    low = re.sub(r"\bras\s+ul\b", "ras al", low)
    low = re.sub(r"\bul\b", "al", low)
    low = re.sub(r"\bel\b", "al", low)
    low = normalize_ws(low)

    mapped = PLACE_MAP.get(low)
    if mapped:
        return mapped

    # Normalize "town in the X district" -> "X district (town unknown)"
    m = re.search(r"\btown\s+in\s+(?:the\s+)?([a-z][a-z \-']{2,40})\s+district\b", low)
    if m:
        district = normalize_ws(m.group(1))
        return f"{district.title()} district (town unknown)"

    # Titlecase fallback
    if len(s) <= 3 and s.isupper():
        return s
    return " ".join([w[:1].upper() + w[1:] if w else w for w in low.split(" ")])

def is_valid_place(place: str) -> bool:
    if not place:
        return False
    p = normalize_ws(place)
    if not p:
        return False
    if is_ship(p):
        return True

    # Hard gate: avoid clause fragments being treated as places
    tokens = [t for t in re.split(r"\s+", p) if t]
    if len(tokens) > MAX_PLACE_WORDS:
        return False

    low = p.lower()
    # Reject clause-like strings when they contain verb-ish tokens and are multi-word
    if BAD_PLACE_TOKEN_PAT.search(low) and len(tokens) >= 4:
        return False

    # Digits are almost always noise in this corpus
    if re.search(r"\d", p):
        return False

    if low in NON_GEO_PLACEHOLDERS:
        return False
    if low in ("there", "here", "this place", "that place"):
        return False
    return True

# ------------------- DATE PARSING (CONSERVATIVE) -------------------
# ------------------- DATE PARSING (CONSERVATIVE) -------------------
ISO_DATE_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")

def extract_year_from_document_date(doc_date: str):
    if not doc_date:
        return None
    m = re.search(r"\b(17|18|19|20)\d{2}\b", doc_date)
    return int(m.group(0)) if m else None

def parse_day_month(text: str):
    if not text:
        return None
    s = normalize_ws(text.strip().lower().replace(",", " "))
    m = re.search(r"\b(\d{1,2})(st|nd|rd|th)?\s+([a-z]+)\b", s)
    if not m:
        return None
    day = int(m.group(1))
    mon_name = m.group(3)
    for k in MONTHS:
        if mon_name.startswith(k[:3]):
            return day, MONTHS[k]
    return None

def to_iso_date(date_str: str, doc_year: int = None, allow_derive_year: bool = True):
    if not date_str:
        return (None, "unknown")
    s = normalize_ws(date_str)

    if ISO_DATE_PAT.match(s):
        return (s, "explicit")

    # D/8-12-28 (dd-mm-yy)
    m = re.search(r"(?:\bD/?\s*)?(\d{1,2})-(\d{1,2})-(\d{2})\b", s, flags=re.I)
    if m and doc_year and allow_derive_year:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        century = (doc_year // 100) * 100
        year = century + yy
        if abs(year - doc_year) <= 10 and 1 <= mm <= 12 and 1 <= dd <= 31:
            return (f"{year:04d}-{mm:02d}-{dd:02d}", "derived_from_doc")

    dm = parse_day_month(s)
    if dm and doc_year and allow_derive_year:
        dd, mm = dm
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return (f"{doc_year:04d}-{mm:02d}-{dd:02d}", "derived_from_doc")

    return (s, "unknown")

# ------------------- PROMPTS -------------------
EXTRACT_PROMPT = """You are an information extraction engine working on historical OCR documents about slavery/manumission in the Persian Gulf region.

GOAL
Extract ONLY enslaved persons (slaves) mentioned in the OCR text and the places each enslaved person is explicitly stated to have been at or moved between.
Your output will be STRICTLY VALIDATED. Do NOT guess, do NOT invent, and do NOT force travel edges.

COMPLETENESS REQUIREMENT
You MUST scan the ENTIRE OCR text and extract ALL place-linked events for each enslaved person.
Common missed items you MUST include when stated:
- "inhabitant of X" (treat as presence at X)
- "lived in X village / in Y district" (treat as presence at that village/district label)
- chained moves: "went to X and thence to Y and from there arrived at Z" (record X, Y, Z in order)
- "we/us reached X", "arrived at X", "sent to X", "took me/us to X", "landed at X"

CRITICAL FILTERING
- Include a person ONLY if the text explicitly states OR strongly implies they are enslaved (sold, bought, captured/kidnapped, described as slave, in custody as slave, manumitted).
- Exclude non-slaves: masters/owners, buyers/sellers, officials, rulers/sheikhs, witnesses, generic roles.

ENSLAVED EVIDENCE (required)
For each included person, provide one short quote (<=25 words) proving enslavement/manumission/capture/ownership context.

PLACE / TRAVEL EXTRACTION RULES (accuracy first)
You will output *events*.
- Record a place ONLY when the text explicitly links the enslaved person to that place (captured in X, taken to Y, lived in Z, arrived at A, reached B, sold in C, etc.).
- Do NOT attach context-only places (cholera at X, letter from Y, official stationed at Z) unless the enslaved person is explicitly at that place.
- Do NOT output vague placeholders as places (e.g., "diving location", "at sea"). Only output a named geographic place or a ship name (H.M.S., S.S., steamer, dhow).

GROUP / PRONOUN RULE (for statements)
If the text explicitly groups the enslaved person with others (e.g., "kidnapped me and my children", "we/us"),
then movements stated for "we/us" may be recorded for that enslaved person.
Do NOT apply those movements to someone who was explicitly separated (sold/captured) before the move.

EVENT TYPES
- presence: person explicitly at a place (inhabitant of, lived in, captured in, found in, sold in).
- movement: explicit movement with BOTH origin and destination stated (from X to Y; brought from X to Y; returned from X to Y).
- stay: explicit residence/stay at a place (duration may be in evidence).
- manumission: manumission/certificate issued/received for the person (record place only if stated).
- death_report: death reported at a place (only if the place is explicitly stated).

IMPORTANT PLACE RULE
- For any non-movement event, place MUST be a non-empty real place string. If the place is not explicitly stated, OMIT the event. Never output place=null for non-movement.
- For type="movement": NEVER output from_place=null or to_place=null. If one endpoint is missing, use presence at the known place instead.

DATES
- If a date is explicitly attached to an event, output ISO YYYY-MM-DD if you can do so confidently; otherwise keep original.
- If no explicit date for an event, set date=null and date_confidence="unknown".
- NEVER invent dates.

EVIDENCE
Every event must include an evidence quote (<=25 words) supporting that event.

OUTPUT (JSON ONLY; no markdown, no extra text)
Schema:
{{
  "doc_id": string,
  "document_date": string|null,
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

Rules for fields:
- For type="movement": use from_place and to_place (both non-null, non-empty). Set place=null.
- For other types: use place (non-null, non-empty). Set from_place=null and to_place=null.

doc_id: "{doc_id}"

OCR TEXT:
<<<{ocr}>>>"""


# A lighter prompt for index/contents pages (performance optimization).
# These pages are usually lists of cases; they rarely contain full narratives.
EXTRACT_PROMPT_INDEX = """You are an information extraction engine working on historical OCR documents.

THIS PAGE IS AN INDEX / LIST (many short entries). Extract ONLY what is explicitly present in each entry.
- Include a person ONLY if the entry explicitly indicates enslavement/manumission (e.g., contains "slave", "manumission", "captured", "bought/sold", "freed").
- For each person: provide enslaved_evidence as the exact entry line (<=25 words; truncate if needed).
- Events: use ONLY presence events when an explicit place name is present in that same entry line. Otherwise OMIT events.
- Do NOT invent places, do NOT infer travel.

OUTPUT JSON ONLY matching the schema, doc_id must be exactly: "{doc_id}".

Schema:
{
  "doc_id": string,
  "document_date": string|null,
  "people": [
    {
      "name": string,
      "enslaved_status": "explicit"|"strong_inferred",
      "enslaved_evidence": string,
      "events": [
        {
          "type": "presence",
          "place": string,
          "from_place": null,
          "to_place": null,
          "date": null,
          "date_confidence": "unknown",
          "evidence": string
        }
      ]
    }
  ]
}

OCR TEXT:
<<<{ocr}>>>"""
REPAIR_PROMPT = """Fix the following so it is ONLY valid JSON matching the schema exactly.
Use doc_id: {doc_id} (must match exactly).
Do NOT add new facts. Prefer DROPPING invalid/unsupported events rather than inventing or keeping schema-breaking content.
If an event has no explicit place, DROP the event (do not use place=null).

SCHEMA:
{{
  "doc_id": string,
  "document_date": string|null,
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

TEXT TO FIX:
<<<{bad}>>>"""

# ------------------- OLLAMA CALLS -------------------
def call_ollama(prompt: str, *, num_predict: int = DEFAULT_NUM_PREDICT, num_ctx: Optional[int] = DEFAULT_NUM_CTX) -> str:
    """Call Ollama /api/generate with retries. Reuses a single HTTP session for speed."""
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

            r = _OLLAMA_SESSION.post(
                OLLAMA_URL,
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            r.raise_for_status()
            # Ollama returns {"response": "...", ...}
            return (r.json().get("response") or "").strip()

        except (ReadTimeout, ConnectionError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SECONDS * attempt
            print(f"  [WARN] Ollama timeout/connection issue (attempt {attempt}/{MAX_CALL_RETRIES}). Retrying in {wait}s...")
            time.sleep(wait)

        except requests.HTTPError as e:
            # Bad request, model not found, etc. Usually not recoverable, but we still retry briefly.
            last_err = e
            wait = min(5 * attempt, 15)
            try:
                detail = r.text[:500]
            except Exception:
                detail = ""
            print(f"  [WARN] Ollama HTTP error (attempt {attempt}/{MAX_CALL_RETRIES}): {e}. {detail} Retrying in {wait}s...")
            time.sleep(wait)

    raise RuntimeError(f"Ollama call failed after {MAX_CALL_RETRIES} retries: {last_err}")


def extract_json(text: str):
    """Best-effort extraction of the first JSON object from a model response."""
    if not text:
        return None

    s = (text or "").strip()

    # Strip common markdown fences
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()

    # Fast path
    try:
        return json.loads(s)
    except Exception:
        pass

    # Balanced-brace scan for the first JSON object
    start = s.find("{")
    if start < 0:
        return None

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except Exception:
                        return None

    # Fallback: last resort greedy regex
    m = re.search(r"\{.*\}", s, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            return None
    return None

# ------------------- VALIDATION / CLEANUP -------------------

ALLOWED_TYPES = {"presence", "movement", "stay", "manumission", "death_report"}
ALLOWED_DATE_CONF = {"explicit", "derived_from_doc", "unknown"}

def validate_strict_reason(obj: dict):
    if not isinstance(obj, dict):
        return False, "not a dict"
    if not isinstance(obj.get("doc_id"), str) or not obj.get("doc_id").strip():
        return False, "missing/invalid doc_id"
    if "people" not in obj or not isinstance(obj["people"], list):
        return False, "missing/invalid people list"

    for i, p in enumerate(obj["people"]):
        if not isinstance(p, dict):
            return False, f"people[{i}] not a dict"
        name = normalize_name(p.get("name", ""))
        if not name:
            return False, f"people[{i}] missing/invalid name"
        if not isinstance(p.get("enslaved_evidence"), str) or not p["enslaved_evidence"].strip():
            return False, f"people[{i}] missing enslaved_evidence"
        if p.get("enslaved_status") not in ("explicit", "strong_inferred"):
            return False, f"people[{i}] invalid enslaved_status"
        if "events" not in p or not isinstance(p["events"], list):
            return False, f"people[{i}] missing/invalid events list"

        for j, ev in enumerate(p["events"]):
            if not isinstance(ev, dict):
                return False, f"people[{i}].events[{j}] not a dict"
            if ev.get("type") not in ALLOWED_TYPES:
                return False, f"people[{i}].events[{j}] invalid type"
            if ev.get("date_confidence") not in ALLOWED_DATE_CONF:
                return False, f"people[{i}].events[{j}] invalid date_confidence"
            if not isinstance(ev.get("evidence"), str) or not ev["evidence"].strip():
                return False, f"people[{i}].events[{j}] missing evidence"

            if ev["type"] == "movement":
                fp = normalize_ws(ev.get("from_place") or "")
                tp = normalize_ws(ev.get("to_place") or "")
                if not fp or not tp:
                    return False, f"people[{i}].events[{j}] movement missing from/to"
                if ev.get("place") not in (None, "", "null"):
                    return False, f"people[{i}].events[{j}] movement must not have place"
            else:
                pl = normalize_ws(ev.get("place") or "")
                if not pl:
                    return False, f"people[{i}].events[{j}] non-movement missing place"
                if ev.get("from_place") not in (None, "", "null"):
                    return False, f"people[{i}].events[{j}] non-movement must not have from_place"
                if ev.get("to_place") not in (None, "", "null"):
                    return False, f"people[{i}].events[{j}] non-movement must not have to_place"

    return True, "ok"

EXPLICIT_TERMS = re.compile(r"\b(sold|bought|slave|captur|kidnap|manumit|owned|belong|purchase)\b", re.I)

def coerce_enslaved_status(status: str, evidence: str) -> str:
    """Downgrade 'explicit' if evidence doesn't actually contain explicit enslavement triggers."""
    if status not in ("explicit", "strong_inferred"):
        return "strong_inferred"
    if status == "explicit":
        if not EXPLICIT_TERMS.search(evidence or ""):
            return "strong_inferred"
    return status

def clean_obj(obj: dict, allow_derive_year: bool = True) -> dict:
    """
    Deterministic cleanup:
    - normalize names and places
    - drop non-personal names
    - drop invalid places
    - normalize/derive ISO dates conservatively (optional)
    - drop schema-inconsistent leftovers
    - dedupe events (stable)
    """
    if not isinstance(obj, dict):
        return obj

    doc_date = obj.get("document_date")
    doc_year = extract_year_from_document_date(doc_date or "")

    cleaned_people = []
    for p in obj.get("people", []):
        name = normalize_name(p.get("name", ""))
        if not is_likely_personal_name(name):
            continue

        enslaved_evidence = normalize_ws(p.get("enslaved_evidence", ""))[:300]
        enslaved_status = coerce_enslaved_status(p.get("enslaved_status"), enslaved_evidence)
        if not enslaved_evidence:
            continue

        events_out = []
        for ev in p.get("events", []):
            if not isinstance(ev, dict):
                continue
            typ = ev.get("type")
            if typ not in ALLOWED_TYPES:
                continue

            evidence = normalize_ws(ev.get("evidence", ""))
            if not evidence:
                continue
            if len(evidence.split()) > 28:
                evidence = " ".join(evidence.split()[:25])

            date_raw = ev.get("date")
            iso_date, conf = to_iso_date(date_raw, doc_year=doc_year, allow_derive_year=allow_derive_year)
            if iso_date is None:
                iso_date = None
                conf = "unknown"
            if conf not in ALLOWED_DATE_CONF:
                conf = "unknown"

            if typ == "movement":
                fp = normalize_place(ev.get("from_place", ""))
                tp = normalize_place(ev.get("to_place", ""))
                if not (is_valid_place(fp) and is_valid_place(tp)):
                    continue
                events_out.append({
                    "type": "movement",
                    "place": None,
                    "from_place": fp,
                    "to_place": tp,
                    "date": iso_date,
                    "date_confidence": conf,
                    "evidence": evidence
                })
            else:
                pl = normalize_place(ev.get("place", ""))
                if not is_valid_place(pl):
                    continue
                events_out.append({
                    "type": typ,
                    "place": pl,
                    "from_place": None,
                    "to_place": None,
                    "date": iso_date,
                    "date_confidence": conf,
                    "evidence": evidence
                })

        # Stable dedupe
        seen = set()
        deduped = []
        for ev in events_out:
            key = (
                ev["type"],
                ev.get("place"),
                ev.get("from_place"),
                ev.get("to_place"),
                ev.get("date"),
                ev.get("evidence"),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(ev)

        if deduped:
            cleaned_people.append({
                "name": name,
                "enslaved_status": enslaved_status,
                "enslaved_evidence": enslaved_evidence,
                "events": deduped
            })

    return {
        "doc_id": str(obj.get("doc_id") or ""),
        "document_date": obj.get("document_date"),
        "people": cleaned_people
    }

def parse_clean_validate(resp_text: str, doc_id: str, derive_year: bool):
    obj = extract_json(resp_text)
    if not isinstance(obj, dict):
        return None, "parse_failed"
    obj["doc_id"] = str(doc_id)
    cleaned = clean_obj(obj, allow_derive_year=derive_year)
    cleaned["doc_id"] = str(doc_id)
    ok, reason = validate_strict_reason(cleaned)
    if not ok:
        return None, reason
    return cleaned, "ok"

# ------------------- RULE-BASED NARRATOR AUGMENTATION -------------------
NARRATOR_PAT = re.compile(
    r"\bstatement\s+made\s+by\s+([A-Za-z][A-Za-z'\- ]{1,60})",
    re.I
)

def detect_narrator_name(ocr: str):
    if not ocr:
        return None
    m = NARRATOR_PAT.search(ocr)
    if not m:
        return None
    chunk = normalize_ws(m.group(1))
    # stop at kinship terms / metadata
    chunk = re.split(r"\b(daughter|son|wife|inhabitant|aged|recorded|resident|of)\b", chunk, flags=re.I)[0]
    name = normalize_name(chunk)
    return name or None

def _sentence_around(text: str, start: int, max_words: int = 25) -> str:
    """Extract a short evidence snippet around a match index."""
    if not text:
        return ""
    # Find sentence boundaries
    left = text.rfind(".", 0, start)
    left_n = text.rfind("\n", 0, start)
    left = max(left, left_n)
    left = 0 if left < 0 else left + 1

    right_dot = text.find(".", start)
    right_nl = text.find("\n", start)
    rights = [r for r in [right_dot, right_nl] if r != -1]
    right = min(rights) if rights else min(len(text), start + 300)

    snippet = normalize_ws(text[left:right])
    words = snippet.split()
    if len(words) > max_words:
        snippet = " ".join(words[:max_words])
    return snippet

# Patterns for explicit place-linked statements in first-person narratives.
# IMPORTANT: patterns are non-greedy and stop at conjunctions/metadata to avoid capturing clause fragments.
PLACE_PATTERNS = [
    ("presence", re.compile(r"\binhabitant\s+of\s+([A-Za-z][A-Za-z'\- ]{2,60}?)(?=\s*(?:,|\.|\baged\b|\brecorded\b|\bon\b|\n))", re.I)),
    ("presence", re.compile(r"\blived\s+in\s+the\s+([A-Za-z][A-Za-z'\- ]{1,50}?)\s+village(?:\s+which\s+is\s+in\s+the\s+([A-Za-z][A-Za-z'\- ]{1,50}?)\s+district)?(?=\s*(?:,|\.|\n))", re.I)),
    ("presence", re.compile(r"\bvillage\s+called\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\bsituated\b|\bwhich\b|\bwho\b|\bon\b|\bin\b|\bat\b|\n))", re.I)),
    ("presence", re.compile(r"\bshipped\s+us\s+off\s+to\s+a\s+town\s+in\s+the\s+([A-Za-z][A-Za-z'\- ]{2,40}?)\s+district\b", re.I)),
    ("presence", re.compile(r"\bwent\s+to\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\bthence\b|\bfrom\b|\bto\s+earn\b|\n))", re.I)),
    ("presence", re.compile(r"\bthence\s+to\s+(?:the\s+)?([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\bfrom\b|\barrived\b|\breached\b|\bto\s+earn\b|\n))", re.I)),
    ("presence", re.compile(r"\barrived\s+at\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\bI\b|\bwe\b|\n))", re.I)),
    ("presence", re.compile(r"\breached\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\bI\b|\bwe\b|\n))", re.I)),
    ("presence", re.compile(r"\bsent\s+us\s+to\s+the\s+village\s+called\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\n))", re.I)),
    ("presence", re.compile(r"\btook\s+me\s+to\s+([A-Za-z][A-Za-z'\- ]{1,50}?)(?=\s*(?:,|\.|\band\b|\n))", re.I)),
]

def extract_narrator_place_hits(ocr: str):
    """
    Return a list of (pos, place, evidence_snippet) for explicit place-linked statements
    in first-person narratives, in document order. This is conservative (presence only).
    """
    hits = []
    if not ocr:
        return hits

    for typ, pat in PLACE_PATTERNS:
        for m in pat.finditer(ocr):
            if pat.pattern.startswith(r"\blived\s+in"):
                village = normalize_ws(m.group(1))
                district = normalize_ws(m.group(2) or "")
                place_raw = f"{village} village" + (f" ({district} district)" if district else "")
            elif "shipped" in pat.pattern:
                district = normalize_ws(m.group(1))
                place_raw = f"{district} district (town unknown)"
            else:
                place_raw = normalize_ws(m.group(1))

            # Hard gate to avoid clause-fragments being treated as place candidates
            if len(place_raw.split()) > MAX_PLACE_WORDS:
                continue
            place = normalize_place(place_raw)
            if not is_valid_place(place):
                continue

            evidence = _sentence_around(ocr, m.start())
            if not evidence:
                continue

            hits.append((m.start(), place, evidence))

    hits.sort(key=lambda x: x[0])
    return hits

def extract_narrator_place_events(ocr: str):
    """Convert narrator hits to presence events (kept for backward compatibility)."""
    events = []
    for _, place, evidence in extract_narrator_place_hits(ocr):
        events.append({
            "type": "presence",
            "place": place,
            "from_place": None,
            "to_place": None,
            "date": None,
            "date_confidence": "unknown",
            "evidence": evidence
        })
    return events

    hits = []
    for typ, pat in PLACE_PATTERNS:
        for m in pat.finditer(ocr):
            if pat.pattern.startswith(r"\blived\s+in"):
                village = normalize_ws(m.group(1))
                district = normalize_ws(m.group(2) or "")
                place_raw = f"{village} village" + (f" ({district} district)" if district else "")
            elif "shipped" in pat.pattern:
                district = normalize_ws(m.group(1))
                place_raw = f"{district} district (town unknown)"
            else:
                place_raw = normalize_ws(m.group(1))

            # Hard gate to avoid clause-fragments being treated as place candidates
            if len(place_raw.split()) > MAX_PLACE_WORDS:
                continue
            place = normalize_place(place_raw)
            if not is_valid_place(place):
                continue

            evidence = _sentence_around(ocr, m.start())
            if not evidence:
                continue

            hits.append((m.start(), {
                "type": "presence",
                "place": place,
                "from_place": None,
                "to_place": None,
                "date": None,
                "date_confidence": "unknown",
                "evidence": evidence
            }))

    hits.sort(key=lambda x: x[0])
    for _, ev in hits:
        events.append(ev)
    return events

# --- helper: fuzzy name match inside a window (handles OCR variants like Hdssun/Hassun) ---
def _name_in_window(window: str, name: str) -> bool:
    """
    Fuzzy name presence check inside a short OCR window.
    Handles small OCR variants (Hdssun vs Hassun) and glued suffixes (Hassunraged).
    """
    if not window or not name:
        return False

    def norm_tokens(s: str):
        # keep letters/apostrophes only; strip common glued descriptors
        toks = re.findall(r"[A-Za-z']+", (s or "").lower())
        out = []
        for t in toks:
            t = re.sub(r"(?:r?aged|about|years?)$", "", t)  # Hassunraged -> hassun
            if t:
                out.append(t)
        return out

    w_tokens = norm_tokens(window)
    n_tokens = norm_tokens(name)
    if not n_tokens:
        return False

    w = " ".join(w_tokens)
    n = " ".join(n_tokens)

    # exact token match
    if re.search(rf"\b{re.escape(n)}\b", w):
        return True

    # fuzzy token match
    for nt in n_tokens:
        if len(nt) < 3:
            continue
        for wt in w_tokens:
            # prefix match (handles hassunraged)
            if (wt.startswith(nt) or nt.startswith(wt)) and min(len(wt), len(nt)) >= 4:
                return True
            if abs(len(wt) - len(nt)) > 4:
                continue
            if difflib.SequenceMatcher(None, wt, nt).ratio() >= 0.80:
                return True

    return False

def augment_with_narrator_rules(obj: dict, ocr: str, derive_year: bool):
    """
    Conservative completeness pass for first-person narratives ("Statement made by ...").

    - Adds narrator place-hits as ordered presence events on the narrator.
    - Adds conservative stay events when the text says "kept us ... godown/house ...".
    - Propagates the narrator's ordered place-hits to named enslaved children (son/daughter/children named/infant born)
      until the first explicit separation mention for that child.

    This pass never invents locations; it only uses explicit, evidenced place mentions already present in the OCR.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("people"), list):
        return obj

    ocr = ocr or ""
    narrator = detect_narrator_name(ocr)
    if not narrator:
        return obj

    hits = extract_narrator_place_hits(ocr)
    if not hits:
        return obj

    def _truncate_words(s: str, n: int = 25) -> str:
        w = normalize_ws(s).split()
        return " ".join(w[:n]) if len(w) > n else normalize_ws(s)

    # Group-membership snippet (used only as evidence prefix for propagated child events)
    kidnap_m = re.search(r"kidnap[^.\n]{0,200}(?:me|us)[^.\n]{0,140}children", ocr, flags=re.I)
    group_snip = _truncate_words(_sentence_around(ocr, kidnap_m.start()) if kidnap_m else "", 18)

    def birth_pos(child_name: str) -> int:
        if not child_name:
            return -1
        n = re.escape(child_name)
        m = re.search(rf"gave\s+birth[^.\n]{{0,220}}\b{n}\b", ocr, flags=re.I)
        return m.start() if m else -1

    def is_narrator_child(child_name: str) -> bool:
        """True if OCR explicitly links this name as narrator's child (son/daughter/child/children named/infant born)."""
        if not child_name:
            return False

        # Birth mention implies child
        if birth_pos(child_name) != -1:
            return True

        kin_windows = []
        for m in re.finditer(r"(my\s+(?:son|daughter|child|children|sons|daughters)[^.\n]{0,220})", ocr, flags=re.I):
            kin_windows.append(ocr[m.start():m.end()])
        for m in re.finditer(r"(children\s+named[^.\n]{0,260})", ocr, flags=re.I):
            kin_windows.append(ocr[m.start():m.end()])
        for m in re.finditer(r"(infant\s+(?:called|named)[^.\n]{0,140})", ocr, flags=re.I):
            kin_windows.append(ocr[m.start():m.end()])

        for win in kin_windows:
            if _name_in_window(win, child_name):
                return True
        return False

    def first_sep_pos(child_name: str) -> int:
        """
        Find first separation point for a child:
        - sold/captured/took ... CHILD
        - must be about a child (kinship cue in same clause) to avoid narrator-only actions like "sold me"
        """
        if not child_name:
            return -1

        poss = []
        clause_specs = [
            ("sold", re.compile(r"sold[^.\n]{0,320}", re.I)),
            ("captur", re.compile(r"captur[^.\n]{0,320}", re.I)),
            ("took", re.compile(r"took[^.\n]{0,260}", re.I)),
        ]

        for verb, pat in clause_specs:
            for mm in pat.finditer(ocr):
                win = ocr[mm.start():mm.end()]

                # Skip narrator-only actions
                if verb == "sold" and re.search(r"\bsold\s+me\b", win, flags=re.I):
                    continue
                if verb == "took" and re.search(r"\btook\s+me\b", win, flags=re.I):
                    continue

                # Require kinship cue for sold/captured (prevents newborn false separations)
                if verb in ("sold", "captur") and not re.search(r"\b(son|daughter|child|children|sons|daughters)\b", win, flags=re.I):
                    continue

                if _name_in_window(win, child_name):
                    poss.append(mm.start())
                    break

        return min(poss) if poss else -1

    # Build ordered narrator events from hits (and interleaved stay events).
    # We treat narrator hits as the most reliable ordering for these statements.
    presence_items = [(pos, {
        "type": "presence",
        "place": place,
        "from_place": None,
        "to_place": None,
        "date": None,
        "date_confidence": "unknown",
        "evidence": ev_snip
    }) for pos, place, ev_snip in hits]

    # Interleave "stay" events when OCR says "kept us ... godown/house ...", attached to the most recent prior hit place.
    stay_items = []
    for sm in re.finditer(r"kept\s+us[^.\n]{0,240}\b(godown|house)\b[^.\n]{0,140}", ocr, flags=re.I):
        spos = sm.start()
        last_place = None
        for hpos, hplace, _ in hits:
            if hpos <= spos:
                last_place = hplace
            else:
                break
        if last_place:
            stay_items.append((spos, {
                "type": "stay",
                "place": last_place,
                "from_place": None,
                "to_place": None,
                "date": None,
                "date_confidence": "unknown",
                "evidence": _truncate_words(_sentence_around(ocr, spos), 25),
            }))

    narrator_items = sorted(presence_items + stay_items, key=lambda x: x[0])

    # Compress consecutive duplicates by place (presence/stay back-to-back on same place)
    def compress_items(items):
        out = []
        last_place = None
        for _, ev in items:
            pl = ev.get("place")
            if pl and last_place == pl:
                continue
            out.append(ev)
            last_place = pl
        return out

    narrator_events_ordered = compress_items(narrator_items)
    hit_places_set = {ev["place"] for ev in narrator_events_ordered if ev.get("place")}

    # 1) Ensure narrator exists in obj; if missing, create a minimal entry when there is clear enslaved evidence.
    narrator_idx = None
    for idx, p in enumerate(obj["people"]):
        pname = normalize_name(p.get("name", ""))
        if not pname:
            continue
        if pname.lower() == narrator.lower() or narrator.lower().startswith(pname.lower()) or pname.lower().startswith(narrator.lower()):
            narrator_idx = idx
            break

    if narrator_idx is None:
        ev_m = re.search(r"(sold\s+me|kidnap[^.\n]{0,160}(?:me|us)|manumit[^.\n]{0,160}(?:me|us))", ocr, flags=re.I)
        if ev_m:
            ensl_ev = _truncate_words(_sentence_around(ocr, ev_m.start()), 25)
            obj["people"].append({
                "name": normalize_name(narrator),
                "enslaved_status": "strong_inferred",
                "enslaved_evidence": ensl_ev,
                "events": []
            })
            narrator_idx = len(obj["people"]) - 1

    # 2) Replace/merge narrator events
    if narrator_idx is not None:
        p = obj["people"][narrator_idx]
        existing = list(p.get("events") or [])

        # Drop existing undated presence/stay events that refer to places we are adding (we add them in better order).
        kept = []
        for ev in existing:
            typ = ev.get("type")
            pl = ev.get("place")
            fp = ev.get("from_place")
            tp = ev.get("to_place")
            pl_n = normalize_place(pl) if pl else None
            fp_n = normalize_place(fp) if fp else None
            tp_n = normalize_place(tp) if tp else None
            if typ in ("presence", "stay") and (ev.get("date") in (None, "", "null")) and pl_n in hit_places_set:
                continue
            if typ == "movement" and (ev.get("date") in (None, "", "null")) and fp_n in hit_places_set and tp_n in hit_places_set:
                continue
            kept.append(ev)

        p["events"] = narrator_events_ordered + kept

    # 3) Propagate narrator ordered presence hits to named enslaved children
    # (we propagate presence hits only; stay is narrator-specific unless the child is explicitly kept too)
    presence_hits = [(pos, place, ev_snip) for (pos, place, ev_snip) in hits]

    for p in obj["people"]:
            pname = normalize_name(p.get("name", ""))
            if not pname:
                continue
            if narrator_idx is not None:
                narr_name_norm = normalize_name(obj["people"][narrator_idx].get("name", ""))
                if pname.lower() == narr_name_norm.lower():
                    continue

            if not is_narrator_child(pname):
                continue

            start = 0
            bpos = birth_pos(pname)
            if bpos != -1:
                # Include the location the child is born in: last explicit hit at/just before the birth mention.
                last_before = None
                for hpos, _, _ in presence_hits:
                    if hpos <= bpos:
                        last_before = hpos
                    else:
                        break
                start = last_before if last_before is not None else bpos

            end = first_sep_pos(pname)
            if end == -1:
                # If no explicit separation found, allow propagation only if "remained with ..." mentions the child,
                # or if the child is born in-text (infant) with no later separation.
                n_esc = re.escape(pname)
                together = re.search(rf"(remained\s+with|with\s+my\s+sons|with\s+my\s+children)[^.\n]{{0,300}}\b{n_esc}\b", ocr, flags=re.I)
                if together or bpos != -1:
                    end = len(ocr)
                else:
                    continue

            propagated = []
            for hpos, place, ev_snip in presence_hits:
                if hpos < start or hpos > end:
                    continue
                ev_evidence = _truncate_words(f"{group_snip} / {ev_snip}", 25) if group_snip else ev_snip
                propagated.append({
                    "type": "presence",
                    "place": place,
                    "from_place": None,
                    "to_place": None,
                    "date": None,
                    "date_confidence": "unknown",
                    "evidence": ev_evidence
                })

            if not propagated:
                continue

            # Deduplicate consecutive places
            compact = []
            last = None
            for ev in propagated:
                if ev["place"] == last:
                    continue
                compact.append(ev)
                last = ev["place"]

            prop_places = {ev["place"] for ev in compact}

            existing = list(p.get("events") or [])
            kept = []
            for ev in existing:
                typ = ev.get("type")
                pl = ev.get("place")
                pl_n = normalize_place(pl) if pl else None
                if typ in ("presence", "stay") and (ev.get("date") in (None, "", "null")) and pl_n in prop_places:
                    continue
                kept.append(ev)

            p["events"] = compact + kept

    # Re-clean & validate
    merged = clean_obj(obj, allow_derive_year=derive_year)
    ok, _ = validate_strict_reason(merged)
    return merged if ok else obj


# ------------------- IMPORTED SLAVE AUGMENTATION -------------------
# Some statements explicitly describe ancestors/relatives as "imported from X" with slave context.
# Local models sometimes miss adding these people even though they have a real multi-place journey.
IMPORTED_PAT = re.compile(
    r"\b([A-Z][A-Za-z'\- ]{1,70}?)\s+(?:was\s+)?imported\b[^.]{0,220}?\bfrom\s+([A-Za-z][A-Za-z'\- ]{2,50})",
    re.I
)



def _find_known_place_near(text: str, center_idx: int, window: int = 260):
    """
    Find a nearby explicit place mention (from PLACE_MAP keys) close to a match.
    Prefer a mention AFTER center_idx within the window; if none, use the closest BEFORE center_idx.
    This prevents accidentally picking far-away places later in the narrative.
    """
    if not text:
        return None
    lo = max(0, center_idx - window)
    hi = min(len(text), center_idx + window)
    low = text.lower()

    # search after center
    best_after = None
    for k in PLACE_MAP.keys():
        if not k or len(k) < 3:
            continue
        pos = low.find(k, center_idx, hi)
        if pos != -1 and (best_after is None or pos < best_after[0]):
            best_after = (pos, k)

    if best_after:
        return normalize_place(best_after[1])

    # search before center (closest to center)
    best_before = None
    for k in PLACE_MAP.keys():
        if not k or len(k) < 3:
            continue
        pos = low.rfind(k, lo, center_idx)
        if pos != -1:
            dist = center_idx - pos
            if best_before is None or dist < best_before[0]:
                best_before = (dist, k)

    if best_before:
        return normalize_place(best_before[1])
    return None


IMPORTED_PAT2 = re.compile(
    r"\b([A-Z][A-Za-z'\- ]{1,70}?)\s*,\s*imported\b[^.]{0,220}?\bfrom\s+([A-Za-z][A-Za-z'\- ]{2,50})",
    re.I
)


def _find_known_place_after(text: str, start_idx: int):
    """Find the earliest known place mention (from PLACE_MAP keys) after a given index."""
    if not text:
        return None
    low = text.lower()
    best = None  # (pos, key)
    for k in PLACE_MAP.keys():
        if not k or len(k) < 3:
            continue
        pos = low.find(k, start_idx)
        if pos != -1 and (best is None or pos < best[0]):
            best = (pos, k)
    if best:
        return normalize_place(best[1])
    return None

def augment_with_imported_slaves(obj: dict, ocr: str, derive_year: bool):
    """
    Add missing imported enslaved persons when OCR explicitly states:
      <Name> was imported ... from <Place> ... (and nearby mentions "slave")
    Also supports the common OCR punctuation form:
      <Name>, imported from <Place>, ...
    Create a movement event if both from_place and an explicit destination place are present nearby.
    Never invent destinations; if destination can't be found, skip.
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("people"), list):
        return obj
    ocr = ocr or ""
    if not ocr:
        return obj

    existing_names = {normalize_name(p.get("name", "")).lower() for p in obj.get("people", []) if isinstance(p, dict)}
    added_any = False

    for pat in (IMPORTED_PAT, IMPORTED_PAT2):
        for m in pat.finditer(ocr):
            raw_name = normalize_name(m.group(1))

            # Strip leading kinship prefixes that OCR often includes in the "name" slot
            raw_name = re.sub(r"^(?:my\s+)?(?:father|mother)\s+", "", raw_name, flags=re.I)
            raw_name = re.sub(r"^(?:the\s+said)\s+", "", raw_name, flags=re.I)
            raw_name = normalize_name(raw_name)

            raw_from = normalize_ws(m.group(2))
            if not raw_name or not is_likely_personal_name(raw_name):
                continue

            evidence = _sentence_around(ocr, m.start())
            if not evidence:
                continue

            # Require explicit slave context nearby to avoid importing free migrants
            if not re.search(r"\bslave\b", evidence, flags=re.I):
                win = normalize_ws(ocr[max(0, m.start()-180):min(len(ocr), m.end()+320)])
                if not re.search(r"\bslave\b", win, flags=re.I):
                    continue
                evidence = win

            from_place = normalize_place(raw_from)
            if not is_valid_place(from_place):
                continue

            # Try to find an explicit destination place AFTER the "imported" phrase
            to_place = _find_known_place_near(ocr, m.start())
            if to_place and normalize_place(to_place) == normalize_place(from_place):
                to_place = _find_known_place_near(ocr, m.end())

            if not (to_place and is_valid_place(to_place) and normalize_place(to_place) != normalize_place(from_place)):
                continue

            ev = {
                "type": "movement",
                "place": None,
                "from_place": from_place,
                "to_place": to_place,
                "date": None,
                "date_confidence": "unknown",
                "evidence": " ".join(evidence.split()[:25])
            }

            person_key = raw_name.lower()
            if person_key in existing_names:
                for p in obj["people"]:
                    if normalize_name(p.get("name", "")).lower() == person_key:
                        p.setdefault("events", [])
                        p["events"].append(ev)
                        if not EXPLICIT_TERMS.search(p.get("enslaved_evidence", "") or ""):
                            p["enslaved_evidence"] = " ".join(evidence.split()[:25])
                            p["enslaved_status"] = "explicit" if "slave" in evidence.lower() else p.get("enslaved_status", "strong_inferred")
                        break
            else:
                obj["people"].append({
                    "name": raw_name,
                    "enslaved_status": "explicit" if "slave" in evidence.lower() else "strong_inferred",
                    "enslaved_evidence": " ".join(evidence.split()[:25]),
                    "events": [ev]
                })
                existing_names.add(person_key)
            added_any = True

    if added_any:
        merged = clean_obj(obj, allow_derive_year=derive_year)
        ok, _ = validate_strict_reason(merged)
        return merged if ok else obj
    return obj


def improve_enslaved_evidence(obj: dict, ocr: str):
    """
    If a person's enslaved_evidence doesn't indicate slavery/manumission/capture,
    replace it with a better explicit snippet from the OCR:
      - Prefer a sentence containing the person's name + explicit slavery terms
      - Else fall back to a document-level explicit sentence (often "sell us / slaves")
    """
    if not isinstance(obj, dict) or not isinstance(obj.get("people"), list) or not ocr:
        return obj

    chunks = [normalize_ws(s) for s in re.split(r"[.\n]+", ocr) if normalize_ws(s)]
    explicit_sents = [s for s in chunks if EXPLICIT_TERMS.search(s)]
    if not explicit_sents:
        return obj

    fallback = None
    for s in explicit_sents:
        if re.search(r"\b(we|us|our)\b", s, flags=re.I):
            fallback = s
            break
    if fallback is None:
        fallback = explicit_sents[0]

    def trunc(s):
        return " ".join((s or "").split()[:25])

    for p in obj["people"]:
        ev = p.get("enslaved_evidence", "") or ""
        if EXPLICIT_TERMS.search(ev):
            continue
        name = normalize_name(p.get("name", ""))
        if not name:
            continue

        best = None
        for s in explicit_sents:
            if _name_in_window(s, name):
                best = s
                break

        p["enslaved_evidence"] = trunc(best or fallback)

    return obj

# ------------------- TIMELINE BUILDING -------------------
def stable_sort_events_by_date(events, allow_date_conf=("explicit", "derived_from_doc")):
    dated = []
    dated_positions = []
    for i, ev in enumerate(events):
        d = ev.get("date")
        conf = ev.get("date_confidence")
        if d and ISO_DATE_PAT.match(d) and conf in allow_date_conf:
            dated.append((d, i, ev))
            dated_positions.append(i)

    dated_sorted = [ev for _, _, ev in sorted(dated, key=lambda x: (x[0], x[1]))]
    out = list(events)
    it = iter(dated_sorted)
    for pos in dated_positions:
        out[pos] = next(it)
    return out

def events_to_place_timeline(events):
    places = []
    for ev in events:
        if ev["type"] == "movement":
            for pl in (ev["from_place"], ev["to_place"]):
                if pl and (not places or places[-1] != pl):
                    places.append(pl)
        else:
            pl = ev.get("place")
            if pl and (not places or places[-1] != pl):
                places.append(pl)
    return places

def make_text_report(obj: dict) -> str:
    lines = []
    lines.append(f"doc_id: {obj.get('doc_id')}")
    lines.append(f"document_date: {obj.get('document_date')}")
    lines.append("")
    people = obj.get("people", [])
    lines.append(f"People extracted: {len(people)}")
    lines.append("")

    for p in people:
        name = p["name"]
        enslaved_status = p.get("enslaved_status") or "∅"
        enslaved_evidence = (p.get("enslaved_evidence") or "").replace("\n", " ").strip() or "∅"
        lines.append(f"- Name: {name} | enslaved_status: {enslaved_status}")
        lines.append(f'  enslaved_evidence: "{enslaved_evidence}"')

        events = stable_sort_events_by_date(p.get("events", []))
        timeline = events_to_place_timeline(events)
        lines.append("  places_timeline: " + (" → ".join(timeline) if timeline else "(none)"))
        lines.append("  events:")
        for ev in events:
            d = ev.get("date") or "∅"
            dc = ev.get("date_confidence") or "∅"
            if ev["type"] == "movement":
                lines.append(f"    - movement: {ev['from_place']} → {ev['to_place']} | date: {d} ({dc})")
            else:
                lines.append(f"    - {ev['type']}: {ev['place']} | date: {d} ({dc})")
            lines.append(f'      evidence: "{ev.get("evidence","").strip()}"')
        lines.append("")

    return "\n".join(lines)

# ------------------- MAIN -------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="typed_english_text_glmocr")
    ap.add_argument("--out_dir", default="ocr_text_out_json")
    ap.add_argument("--text_out_dir", default="ocr_text_out_text")
    ap.add_argument("--require_keyword", default="", help="Optional keyword; skip docs without it (case-insensitive).")
    ap.add_argument("--derive_year", action="store_true", help="Derive missing year for day-month dates using document_date year.")
    ap.add_argument("--no_augment_narrator", action="store_true", help="Disable rule-based narrator place augmentation for first-person statements.")

    # Performance / diagnostics
    ap.add_argument("--log_dir", default=DEFAULT_LOG_DIR, help="Directory to write run.log, run_status.csv, run_state.json (default: current dir).")
    ap.add_argument("--num_predict", type=int, default=DEFAULT_NUM_PREDICT, help="Max tokens to generate (Ollama option num_predict).")
    ap.add_argument("--num_ctx", type=int, default=(DEFAULT_NUM_CTX or 0), help="Context length (Ollama option num_ctx). 0 = leave default.")
    ap.add_argument("--index_mode", choices=["auto", "on", "off"], default="auto", help="Index/contents optimization: auto/on/off.")
    ap.add_argument("--index_chunk_lines", type=int, default=70, help="When index_mode != off, split OCR into chunks of this many lines.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.text_out_dir, exist_ok=True)

    logger = _setup_logger(args.log_dir)
    status_csv = os.path.join(args.log_dir, "run_status.csv")

    # CSV header
    fieldnames = [
        "ts",
        "i",
        "n",
        "doc_id",
        "file",
        "status",
        "index_like",
        "chunks",
        "ocr_chars",
        "extract_s",
        "repair_s",
        "total_s",
        "repairs",
        "people",
        "events",
        "reason",
    ]

    files = sorted([fn for fn in os.listdir(args.in_dir) if fn.lower().endswith(".txt")])
    total = len(files)
    state = {
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "in_dir": args.in_dir,
        "out_dir": args.out_dir,
        "text_out_dir": args.text_out_dir,
        "total_files": total,
        "ok": 0,
        "fail": 0,
        "skip": 0,
        "last_file": None,
        "last_status": None,
    }
    _write_state(args.log_dir, state)

    need_header = not os.path.exists(status_csv) or os.path.getsize(status_csv) == 0
    with open(status_csv, "a", newline="", encoding="utf-8") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if need_header:
            writer.writeheader()

        for i, fn in enumerate(files, 1):
            path = os.path.join(args.in_dir, fn)
            doc_id = os.path.splitext(fn)[0]
            out_json_path = os.path.join(args.out_dir, f"{doc_id}.json")
            out_txt_path = os.path.join(args.text_out_dir, f"{doc_id}.txt")
            ts = datetime.datetime.now().isoformat(timespec="seconds")

            doc_start = time.time()
            extract_s = 0.0
            repair_s = 0.0
            repairs = 0
            chunks_used = 1
            index_like = False
            reason = ""
            status = ""

            if os.path.exists(out_json_path):
                status = "skip_exists"
                logger.info(f"[SKIP] {fn} (already processed)")
                writer.writerow({
                    "ts": ts, "i": i, "n": total, "doc_id": doc_id, "file": fn,
                    "status": status, "index_like": False, "chunks": 0,
                    "ocr_chars": 0, "extract_s": 0, "repair_s": 0,
                    "total_s": 0, "repairs": 0, "people": 0, "events": 0,
                    "reason": "already_processed",
                })
                fcsv.flush()
                state["skip"] += 1
                state["last_file"] = fn
                state["last_status"] = status
                _write_state(args.log_dir, state)
                continue

            ocr = open(path, "r", encoding="utf-8", errors="ignore").read()
            if args.require_keyword and (args.require_keyword.lower() not in ocr.lower()):
                status = "skip_keyword"
                logger.info(f"[SKIP] {fn} (missing keyword: {args.require_keyword})")
                writer.writerow({
                    "ts": ts, "i": i, "n": total, "doc_id": doc_id, "file": fn,
                    "status": status, "index_like": False, "chunks": 0,
                    "ocr_chars": len(ocr), "extract_s": 0, "repair_s": 0,
                    "total_s": round(time.time() - doc_start, 3), "repairs": 0,
                    "people": 0, "events": 0, "reason": "missing_keyword",
                })
                fcsv.flush()
                state["skip"] += 1
                state["last_file"] = fn
                state["last_status"] = status
                _write_state(args.log_dir, state)
                continue

            # Decide whether to use index optimization
            if args.index_mode == "on":
                index_like = True
            elif args.index_mode == "off":
                index_like = False
            else:
                index_like = looks_like_index_page(ocr)

            logger.info(f"[RUN ] {i:04d}/{total:04d} {fn} (index_like={index_like}, ocr_chars={len(ocr)})")

            try:
                obj = None
                repaired_text = None

                if index_like:
                    chunks = chunk_text_by_lines(ocr, lines_per_chunk=max(10, args.index_chunk_lines))
                    chunks_used = len(chunks)
                    partials = []
                    failed_chunks = 0

                    for ci, chunk in enumerate(chunks, 1):
                        prompt = EXTRACT_PROMPT_INDEX.format(doc_id=doc_id, ocr=chunk)
                        t0 = time.time()
                        resp = call_ollama(prompt, num_predict=args.num_predict, num_ctx=(args.num_ctx or None))
                        extract_s += (time.time() - t0)

                        obj_i, r_i = parse_clean_validate(resp, doc_id, derive_year=args.derive_year)
                        if obj_i is None:
                            repairs += 1
                            t0 = time.time()
                            repaired_text = call_ollama(REPAIR_PROMPT.format(doc_id=doc_id, bad=resp),
                                                       num_predict=args.num_predict, num_ctx=(args.num_ctx or None))
                            repair_s += (time.time() - t0)
                            obj_i, r_i = parse_clean_validate(repaired_text, doc_id, derive_year=args.derive_year)

                        if obj_i is not None:
                            partials.append(obj_i)
                        else:
                            failed_chunks += 1

                    if partials:
                        obj = merge_partial_objects(partials, doc_id)
                        reason = "ok" if failed_chunks == 0 else f"partial_ok_failed_chunks={failed_chunks}/{chunks_used}"
                    else:
                        obj = None
                        reason = f"all_chunks_failed={failed_chunks}/{chunks_used}"

                else:
                    t0 = time.time()
                    resp = call_ollama(EXTRACT_PROMPT.format(doc_id=doc_id, ocr=ocr),
                                       num_predict=args.num_predict, num_ctx=(args.num_ctx or None))
                    extract_s += (time.time() - t0)

                    obj, reason = parse_clean_validate(resp, doc_id, derive_year=args.derive_year)

                    if obj is None:
                        repairs += 1
                        t0 = time.time()
                        repaired_text = call_ollama(REPAIR_PROMPT.format(doc_id=doc_id, bad=resp),
                                                   num_predict=args.num_predict, num_ctx=(args.num_ctx or None))
                        repair_s += (time.time() - t0)
                        obj, reason = parse_clean_validate(repaired_text, doc_id, derive_year=args.derive_year)

                if obj is None:
                    status = "fail"
                    logger.error(f"[FAIL] {fn} ({reason}; saved raw output)")
                    bad_path = os.path.join(args.out_dir, f"{doc_id}.bad.txt")
                    with open(bad_path, "w", encoding="utf-8") as f:
                        # save the last resp if present, else empty
                        f.write(resp if "resp" in locals() else "")
                    if repaired_text:
                        with open(os.path.join(args.out_dir, f"{doc_id}.bad.repair.txt"), "w", encoding="utf-8") as f:
                            f.write(repaired_text)

                    writer.writerow({
                        "ts": ts, "i": i, "n": total, "doc_id": doc_id, "file": fn,
                        "status": status, "index_like": index_like, "chunks": chunks_used,
                        "ocr_chars": len(ocr), "extract_s": round(extract_s, 3), "repair_s": round(repair_s, 3),
                        "total_s": round(time.time() - doc_start, 3), "repairs": repairs,
                        "people": 0, "events": 0, "reason": reason,
                    })
                    fcsv.flush()
                    state["fail"] += 1
                    state["last_file"] = fn
                    state["last_status"] = status
                    _write_state(args.log_dir, state)
                    continue

                # Optional narrator augmentation (helps completeness on "Statement made by ..." narratives)
                # Skip this on index-like pages (usually not useful and costs extra scanning).
                if (not index_like) and (not args.no_augment_narrator):
                    obj2 = augment_with_narrator_rules(obj, ocr, derive_year=args.derive_year)
                    ok, _ = validate_strict_reason(obj2)
                    if ok:
                        obj = obj2

                # Add missing "imported from X" enslaved persons (genealogical statements)
                obj = augment_with_imported_slaves(obj, ocr, derive_year=args.derive_year)

                # If enslaved_evidence is weak/misassigned, replace with a better explicit snippet from OCR
                obj = improve_enslaved_evidence(obj, ocr)

                # Final deterministic clean pass
                obj = clean_obj(obj, allow_derive_year=args.derive_year)

                # Save JSON
                with open(out_json_path, "w", encoding="utf-8") as f:
                    json.dump(obj, f, ensure_ascii=False, indent=2)

                # Save text-only report
                with open(out_txt_path, "w", encoding="utf-8") as f:
                    f.write(make_text_report(obj))

                people_n = len(obj.get("people") or [])
                events_n = sum(len(p.get("events") or []) for p in (obj.get("people") or []))

                status = "ok"
                logger.info(f"[OK  ] {fn} -> {doc_id}.json + {doc_id}.txt  (people={people_n}, events={events_n}, extract={extract_s:.1f}s, repair={repair_s:.1f}s)")

                writer.writerow({
                    "ts": ts, "i": i, "n": total, "doc_id": doc_id, "file": fn,
                    "status": status, "index_like": index_like, "chunks": chunks_used,
                    "ocr_chars": len(ocr), "extract_s": round(extract_s, 3), "repair_s": round(repair_s, 3),
                    "total_s": round(time.time() - doc_start, 3), "repairs": repairs,
                    "people": people_n, "events": events_n, "reason": reason,
                })
                fcsv.flush()
                state["ok"] += 1
                state["last_file"] = fn
                state["last_status"] = status
                _write_state(args.log_dir, state)

            except Exception as e:
                status = "error"
                logger.exception(f"[ERR ] {fn} -> {type(e).__name__}: {e}")
                writer.writerow({
                    "ts": ts, "i": i, "n": total, "doc_id": doc_id, "file": fn,
                    "status": status, "index_like": index_like, "chunks": chunks_used,
                    "ocr_chars": len(ocr), "extract_s": round(extract_s, 3), "repair_s": round(repair_s, 3),
                    "total_s": round(time.time() - doc_start, 3), "repairs": repairs,
                    "people": 0, "events": 0, "reason": f"{type(e).__name__}: {e}",
                })
                fcsv.flush()
                state["fail"] += 1
                state["last_file"] = fn
                state["last_status"] = status
                _write_state(args.log_dir, state)
                continue

    logger.info(f"Done. ok={state['ok']} skip={state['skip']} fail={state['fail']} total={state['total_files']}")

if __name__ == "__main__":
    main()
