#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model-first OCR extraction pipeline for historical slavery / manumission pages.

Design:
- Treat each .txt file as one standalone page.
- Let the model do the substantive classification and extraction work.
- Use deterministic code only to normalize, validate, deduplicate, and write outputs.
- Write outputs incrementally after every page.

Outputs:
- Detailed info.csv
- name place.csv
- run_status.csv
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import logging
import os
import pathlib
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from requests.exceptions import ConnectionError, ReadTimeout


# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://ollama:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "mistral-small3.1:latest")
DEFAULT_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1200"))
DEFAULT_NUM_CTX = os.environ.get("OLLAMA_NUM_CTX")
DEFAULT_NUM_CTX = int(DEFAULT_NUM_CTX) if DEFAULT_NUM_CTX and DEFAULT_NUM_CTX.isdigit() else None
REQUEST_TIMEOUT = (10, 600)
MAX_CALL_RETRIES = 3
RETRY_BACKOFF_SECONDS = 10


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------
DETAIL_COLUMNS = [
    "Name",
    "Page",
    "Report Type",
    "Crime Type",
    "Whether abuse",
    "Conflict Type",
    "Trial",
    "Amount paid",
]

PLACE_COLUMNS = [
    "Name",
    "Page",
    "Place",
    "Order",
    "Arrival Date",
    "Date Confidence",
    "Time Info",
]

STATUS_COLUMNS = [
    "page",
    "filename",
    "status",
    "named_people",
    "detail_rows",
    "place_rows",
    "model_calls",
    "repair_calls",
    "elapsed_seconds",
    "note",
]

REPORT_TYPES = {
    "statement",
    "transport/admin",
    "correspondence",
}

LEGACY_REPORT_TYPE_MAP = {
    "investigation/correspondence": "correspondence",
    "official correspondence": "correspondence",
}

CRIME_TYPES = {
    "kidnapping",
    "sale",
    "trafficking",
    "illegal detention",
    "forced transfer",
    "debt-claim transfer",
}

CONFLICT_TYPES = {
    "manumission dispute",
    "ownership dispute",
    "debt dispute",
    "free-status dispute",
    "forced-transfer dispute",
    "repatriation dispute",
    "kidnapping case",
}

TRIAL_TYPES = {
    "manumission requested",
    "manumission certificate requested",
    "manumission recommended",
    "manumission granted",
    "free status confirmed",
    "released",
    "repatriation arranged",
    "certificate delivered",
}

DATE_CONFIDENCE = {"explicit", "derived_from_doc", "unknown", ""}
STATUS_VALUES = {
    "ok",
    "skip:index",
    "skip:record_metadata",
    "skip:bad_ocr",
    "no_named_people",
    "error",
}

NAME_STOPWORDS = {
    "slave", "slaves", "woman", "man", "boy", "girl", "unknown", "unnamed",
    "agency", "resident", "secretary", "captain", "major", "sheikh", "shaikh",
    "political", "residency", "certificate", "statement", "memorandum", "telegram",
    "master", "owner", "buyer", "seller", "agent", "office",
}

PLACE_STOPWORDS = {
    "unknown", "unclear", "none", "nil", "there", "here", "this agency",
    "the agency", "agency", "residency", "political agency", "residency agency",
    "office", "record", "statement", "memorandum", "certificate", "arrival",
}

PLACE_MAP = {
    "shargah": "Sharjah",
    "sharjeh": "Sharjah",
    "sharjah": "Sharjah",
    "dibai": "Dubai",
    "debai": "Dubai",
    "dubai": "Dubai",
    "bahrein": "Bahrain",
    "bahrain": "Bahrain",
    "bushire": "Bushehr",
    "busheir": "Bushehr",
    "bushehr": "Bushehr",
    "mekran": "Mekran",
    "mokran": "Mekran",
    "henjam": "Henjam",
    "honjam": "Henjam",
    "ras ul khaimah": "Ras al Khaimah",
    "ras al khaimah": "Ras al Khaimah",
    "umm al quwain": "Umm al Quwain",
    "umm ul quwain": "Umm al Quwain",
    "muscat": "Muscat",
    "mascat": "Muscat",
    "zanzibar": "Zanzibar",
    "abyssinia": "Abyssinia",
    "abisinia": "Abyssinia",
}

MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
}

ISO_DATE_PAT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
PAGE_CLASSIFY_PROMPT = """You are reading ONE OCR page from a historical slavery / manumission archive.

Your task is to decide whether this page should be extracted, and if so infer the page-level report type.

Return JSON only:
{
  "should_extract": true,
  "skip_reason": null,
  "report_type": "statement",
  "evidence": "..."
}

Allowed skip_reason values:
- null
- index
- record_metadata
- bad_ocr

Allowed report_type values:
- statement
- transport/admin
- correspondence

Definitions:
- statement: a recorded testimony, declaration, or first-person account by a subject or witness
- transport/admin: the page's main purpose is handling movement, repatriation, passage, maintenance, routing, subsistence, certificate delivery, reimbursement, or other case administration
- correspondence: an official communication page whose main purpose is discussion, recommendation, forwarding, inquiry, or notification, rather than transport/admin case handling

Critical rule:
- Classify by the MAIN FUNCTION of the page, not by document form.
- A telegram, letter, or memo can still be transport/admin if its main purpose is repatriation, passage, maintenance, routing, delivery, or other case handling.
- Only use correspondence when the page is mainly communicative/discussive and not mainly handling transport/admin matters.

Decision hints:
- Choose statement for "Statement of...", "I was born...", "I request...", recorded testimony, or declarations.
- Choose transport/admin for pages about repatriation requests/arrangements, passage, delivery to a place, being taken to a place, subsistence, provisions, maintenance charges, or certificate handling.
- Choose correspondence for general office communication, recommendations, investigative discussion, forwarding notes, and updates that do not mainly handle transport/admin logistics.

Skip only when the page is clearly one of these:
- index or list page
- archive metadata / cover / about-this-record page
- OCR too damaged to extract reliably

Important:
- Use ONLY this page.
- Do not decide skip_reason merely because the page is short or administrative.
- Administrative cover letters that still name manumission subjects should still be extracted.
- evidence must be a short quote or phrase from the page, max 25 words.
- Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

NAME_PASS_PROMPT = """You are extracting ALL named enslaved/manumission subjects from ONE OCR page.

Return JSON only:
{
  "named_people": [
    {"name": string, "evidence": string}
  ]
}

Task:
Find every named person on this page who is themselves an enslaved subject, refugee slave, fugitive slave, manumission applicant, certificate recipient, person recommended for manumission, or person whose statement/case/paper is being handled on this page.

Who to INCLUDE:
- people who are themselves the enslaved subject, slave, refugee slave, fugitive slave, manumission applicant, certificate recipient, or grouped subject on this page
- people named in shared lists such as "the following refugee slaves", "for delivery to:", numbered subject lists, grouped recommendation pages, grouped certificate pages, or grouped forwarding pages
- people on correspondence pages when the page is clearly about granting, recommending, delivering, or forwarding their manumission-related papers
- a child, mother, son, daughter, or grouped family member when the page explicitly gives that person a name and clearly treats them as part of the subject group

Who to EXCLUDE:
- owners, masters, buyers, sellers, captains, rulers, sheikhs, secretaries, agents, clerks, witnesses, correspondents, office staff, and other non-subject people
- a person mentioned only because the page says "sold to X", "sold me to X", "bought by Y", "belonging to Z", "letter from Z", "statement recorded by A", or similar non-subject roles
- any free-born person or any person explicitly described as not being a slave

Critical disambiguation:
- if a statement says the subject was sold TO someone, that buyer is not a subject
- if the page names several owners or masters in a transfer chain, return only the enslaved/manumission subject, not the owners
- do not return a name unless the page supports that this person is part of the enslaved/manumission subject group

How to read the page:
- scan the full page, not just the first paragraph
- look in prose, headings, numbered lines, parenthesized lists, certificate-delivery lines, forwarding lines, and signatures
- if one sentence governs multiple listed names, include every named subject in the list
- if the page is a short office letter but clearly names one or more people whose certificate or case is being handled, include them
- if a name is partly noisy from OCR, return the best page-supported version only; do not invent missing text

Name formatting rules:
- preserve the fullest page-supported personal name string
- keep lineage and kinship connectors such as bin, bint, daughter of, son of, Abu, Umm when present
- do not return titles or office descriptions as names
- include only real named people, not unnamed descriptions like "slave girl", "a woman", "three slaves" with no names

Evidence rules:
- evidence must be a short quote or phrase from the page, max 25 words
- evidence should support why this person is a subject rather than merely proving the name exists

Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

NAME_RECALL_PROMPT = """Do a second, recall-focused pass over this same OCR page and return ONLY named enslaved/manumission subjects that might have been missed.

Return JSON only:
{
  "named_people": [
    {"name": string, "evidence": string}
  ]
}

Look especially for:
- names after "the following refugee slaves"
- names after "for delivery to"
- names in numbered or parenthesized lists
- names in certificate recommendation / delivery pages
- names in grouped correspondence where one sentence applies to several people
- names with lineage forms such as "daughter of" or "son of"
- names appearing after "statement of", "statement made by", "certificate to", "grant certificate to", or "recommend certificate for"

Still exclude owners, officials, buyers/sellers, and any free-born or non-slave person.
If a page contains a transfer chain like "sold to X" or "bought by Y", those names are not subjects.
Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""


NAME_FILTER_PROMPT = """You are filtering candidate names from one extraction stage for ONE OCR page.

You are given:
- the OCR text for one page
- a candidate list produced by a single extraction pass

Return JSON only:
{
  "named_people": [
    {"name": string, "evidence": string}
  ]
}

Task:
Keep ONLY candidate names that are clearly named enslaved subjects, refugee slaves, fugitive slaves, manumission applicants, certificate recipients, or clearly named members of the same enslaved/manumission subject group on this page.

Critical rules:
- Choose ONLY from the provided candidate list. Do not invent new names.
- Exclude owners, masters, buyers, sellers, rulers, sheikhs, captains, correspondents, writers, signatories, witnesses, office staff, and other non-subject people.
- If the page says the subject was sold TO someone, bought by someone, belonged to someone, or was recorded by someone, that other person is not a subject.
- If a short telegram or administrative page explicitly says something like "slave ... named X", "named X ... slave", or "X requests repatriation", keep that person.
- If uncertain, prefer precision: exclude the doubtful name.

Evidence rules:
- evidence must be a short quote or phrase from the page, max 25 words
- evidence should support why this person is a subject rather than merely proving the name exists

Stage: {stage}

CANDIDATE NAMES JSON:
<<<{candidate_names_json}>>>

OCR TEXT:
<<<{ocr}>>>
"""

NAME_VERIFY_PROMPT = """You are doing FINAL name adjudication for ONE OCR page.

You are given:
- the OCR text for one page
- a merged candidate list produced by earlier extraction passes

Return JSON only:
{
  "named_people": [
    {"name": string, "evidence": string}
  ]
}

Task:
From the candidate list, return the COMPLETE final list of named people who are themselves enslaved subjects, refugee slaves, fugitive slaves, manumission applicants, certificate recipients, or clearly named members of the same subject group on this page.

Critical rules:
- Choose ONLY from the provided candidate list. Do not invent new names.
- Return the complete final list, not just changes.
- Keep a relative only when the page clearly treats that named relative as part of the enslaved/manumission subject group.
- Exclude owners, masters, buyers, sellers, rulers, sheikhs, captains, correspondents, writers, signatories, witnesses, office staff, and other non-subject people.
- If the page says the subject was sold TO someone, bought by someone, belonged to someone, or was recorded by someone, that other person is not a subject.
- If a short telegram or administrative page explicitly says something like "slave ... named X", "named X ... slave", or "X requests repatriation", keep that person.
- If uncertain, prefer precision: exclude the doubtful name.

Evidence rules:
- evidence must be a short quote or phrase from the page, max 25 words
- evidence should support why this person is a subject rather than merely proving the name exists

CANDIDATE NAMES JSON:
<<<{candidate_names_json}>>>

OCR TEXT:
<<<{ocr}>>>
"""

META_PASS_PROMPT = """You are extracting person-specific metadata for ONE named enslaved/manumission subject from ONE OCR page.

Target person: {name}
Page number: {page}
Report Type for this page: {report_type}

Report type definitions:
- statement: recorded testimony, declaration, or first-person account
- transport/admin: logistics, expenses, passage, repatriation arrangements, maintenance, reimbursement, certificate handling, or administrative movement
- correspondence: any official letter, telegram, memo, recommendation, forwarding note, or investigative office communication

Return JSON only:
{
  "name": "{name}",
  "page": {page},
  "report_type": "{report_type}",
  "crime_type": null,
  "whether_abuse": "",
  "conflict_type": null,
  "trial": null,
  "amount_paid": null,
  "evidence": {
    "crime_type": null,
    "whether_abuse": null,
    "conflict_type": null,
    "trial": null,
    "amount_paid": null
  }
}

Allowed values:
- crime_type: kidnapping | sale | trafficking | illegal detention | forced transfer | debt-claim transfer | null
- whether_abuse: yes | no | ""
- conflict_type: manumission dispute | ownership dispute | debt dispute | free-status dispute | forced-transfer dispute | repatriation dispute | kidnapping case | null
- trial: manumission requested | manumission certificate requested | manumission recommended | manumission granted | free status confirmed | released | repatriation arranged | certificate delivered | null
- amount_paid: short literal amount string from the page, otherwise null

Rules:
- Use ONLY this page and ONLY the target person.
- Do not copy facts from another person on the same page unless the wording clearly applies to the target person too.
- Leave unsupported fields blank/null rather than guessing.
- whether_abuse = yes only when the page explicitly states beating, cruel treatment, confinement, violence, flogging, starvation, forced prostitution, overwork, threats, prison, chains, or equivalent abuse.
- whether_abuse = no only when the page explicitly says there was no abuse or no ill-treatment.
- crime_type should reflect what happened to the target person on this page.
- conflict_type should reflect the dispute framing of this page, not the target's entire life story.
- trial should reflect the procedural status or outcome on this page only.
- amount_paid should be filled only when the page explicitly gives a payment amount tied to the target person's case handling, maintenance, passage, release, or repatriation.
- each non-null field must have a short supporting evidence quote, max 25 words.
- Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_PASS_PROMPT = """You are extracting CANDIDATE PAGE-LOCAL places for ONE named enslaved/manumission subject from ONE OCR page.

Target person: {name}

Return JSON only:
{
  "name": "{name}",
  "places": [
    {
      "place": string,
      "time_text": string | null,
      "evidence": string
    }
  ]
}

This is a high-recall candidate-finding task.
Use ONLY this page.

Task:
Find every real named place explicitly linked to the target person on this page.

Include:
- birthplace / native place / residence / origin
- kidnapped from / captured from / brought to / taken to / sold at / sent to / arrived at / reached / escaped to / took refuge at
- actual origin, movement, transfer, arrival, refuge, or presence
- any clearly arranged or formally proposed next destination for the target person on this page
- shared places that clearly apply to the target person in grouped name lists or shared administrative lines

Do NOT do these tasks here:
- do NOT assign final route order
- do NOT decide order = 0 versus positive route step
- do NOT infer final arrival_date
- do NOT infer date_confidence from the page date

time_text rules:
- use time_text only for a raw timing phrase directly attached to that place, such as "May 1931", "five years ago", "on arrival", or "about the 17th"
- keep the wording short and literal
- leave time_text null when there is no such attached timing phrase

What NOT to extract:
- ship names
- office titles instead of places
- generic words like agency, residency, office, there, here, sea
- places that belong only to a correspondent, official, or recipient and are not clearly linked to the target person

Evidence rules:
- every place needs a short supporting quote or phrase, max 25 words
- evidence should support the place-target linkage, not final ordering

Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_RECALL_PROMPT = """Do a second, recall-focused pass for PAGE-LOCAL places for the same target person.

Target person: {name}

Return JSON only:
{
  "name": "{name}",
  "places": [
    {
      "place": string,
      "order": integer,
      "arrival_date": string | null,
      "date_confidence": "explicit" | "derived_from_doc" | "unknown",
      "time_text": string | null,
      "evidence": string
    }
  ]
}

Look especially for places often missed:
- birthplace / native place wording
- grouped subject pages where one shared place applies to multiple listed names
- certificate delivery locations
- movement chains like from X to Y and thence to Z
- formal next-step movements such as sent to, sending him to, forwarded to, for delivery to, or repatriation to
- administrative pages where a real forwarding route may exist, e.g. source place plus destination/arrival place
- use order=0 only when the page does not clearly support route membership or target linkage
- dates attached to the place, the handling event, the delivery event, the statement date, or the page date

If a place has no support for a full date, leave arrival_date null and preserve partial timing in time_text.
Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_VERIFY_PROMPT = """You are doing FINAL adjudication of page-local places for ONE named enslaved/manumission subject.

Target person: {name}
Page number: {page}
Candidate place mentions already extracted (high-recall candidates, may include false positives, duplicates, OCR variants, or raw timing text):
{candidate_places_json}

Return JSON only:
{
  "name": "{name}",
  "places": [
    {
      "place": string,
      "order": integer,
      "arrival_date": string | null,
      "date_confidence": "explicit" | "derived_from_doc" | "unknown",
      "time_text": string | null,
      "evidence": string
    }
  ]
}

Task:
Using the OCR text plus the candidate place mentions, produce the COMPLETE FINAL place list for this target person on this page.

Important:
- Start from the candidate list.
- Keep only places that truly belong to the target person's own page-local route, presence, origin, refuge, transfer, arrival, or clearly arranged next movement.
- You may drop wrong candidates, merge duplicates, normalize OCR variants to one place, assign route order, and improve date fields.
- Do not add speculative places.

Core rules:
- Relative, owner, correspondent, writer, or recipient places do not belong to the target unless the page clearly says the target person was also there.
- Ship names are never places.
- "house of X" or "household of X" is not a place unless it clearly names a real settlement/place.
- Generic office words like office, here, there are not places.
- Agency / Political Agency may be kept only when the page clearly says the target person is there or took refuge there.

Order rules:
- Use order = 1,2,3,... only for places that the page clearly supports as part of the target person's route, presence, origin, refuge, transfer, arrival, or formally arranged next movement.
- A place may have positive order even when arrival_date is blank, if the page clearly frames it as the next formal movement step.
- Use order = 0 for relevant places that are administrative-only, background-only, weakly linked, or not clearly part of the route sequence.
- Positive orders must form one consecutive sequence 1..n.

Date and time rules:
- Use arrival_date only when the date is explicitly stated for that place-event, or when it is clearly derivable from the page date for that handling/presence/delivery/recommendation/statement event.
- Use date_confidence = explicit only for directly stated dates.
- Use date_confidence = derived_from_doc only when the page date clearly dates that place-event.
- If the page gives only partial or vague timing, leave arrival_date null and preserve that wording in time_text.
- Do not invent full dates.

Evidence rules:
- Every final place needs a short supporting quote or phrase, max 25 words.
- Evidence should support why the place belongs to the target and, when possible, why its order/date assignment is justified.

Output JSON only.

OCR TEXT:
<<<{ocr}>>>
"""

PLACE_DATE_ENRICH_PROMPT = """You already extracted page-local places for ONE named enslaved/manumission subject from ONE OCR page.

Target person: {name}

Current places:
{places_json}

Your task:
Keep the same places and try to improve ONLY the date-related fields for each place.

Return JSON only:
{
  "name": "{name}",
  "places": [
    {
      "place": string,
      "order": integer,
      "arrival_date": string | null,
      "date_confidence": "explicit" | "derived_from_doc" | "unknown",
      "time_text": string | null,
      "evidence": string
    }
  ]
}

Rules:
- Keep the same place list; do not add new places here.
- Try your best to find a date for each place using ONLY this page.
- Use explicit only when the date is directly stated for that place or event.
- Use derived_from_doc when the page date clearly dates that handling/presence/delivery/recommendation/statement event for that place.
- If the page only gives partial timing such as month-year or vague relative time, leave arrival_date null and put that timing in time_text.
- Do not invent full dates.
- Keep evidence short, max 25 words.

OCR TEXT:
<<<{ocr}>>>
"""

JSON_REPAIR_PROMPT = """Fix the following so it is valid JSON only.
Do not add new facts.

Required top-level shape:
{schema}

TEXT TO FIX:
<<<{bad}>>>
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class CallStats:
    model_calls: int = 0
    repair_calls: int = 0


@dataclass
class PageDecision:
    should_extract: bool
    skip_reason: Optional[str]
    report_type: str
    evidence: str = ""


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------
def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def strip_accents(text: str) -> str:
    if not text:
        return ""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch))


def clean_ocr(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\ufeff", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    cleaned: List[str] = []
    for line in lines:
        line = re.sub(r"[\t ]+", " ", line).strip()
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def page_number_from_path(path: pathlib.Path) -> int:
    digits = re.sub(r"\D", "", path.stem)
    return int(digits) if digits else 0


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})
    tmp.replace(path)


def setup_logger(log_dir: pathlib.Path, verbose: bool) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ner_extract_model_first")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def progress(message: str) -> None:
    print(message, flush=True)


# ---------------------------------------------------------------------------
# Ollama client
# ---------------------------------------------------------------------------
class OllamaClient:
    def __init__(self, url: str, model: str, num_predict: int, num_ctx: Optional[int]) -> None:
        self.url = url
        self.model = model
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.session = requests.Session()
        self.session.headers.update({"Connection": "keep-alive"})

    def generate(self, prompt: str, stats: CallStats, *, num_predict: Optional[int] = None) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(1, MAX_CALL_RETRIES + 1):
            try:
                payload: Dict[str, Any] = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0,
                        "num_predict": int(num_predict or self.num_predict),
                    },
                }
                if self.num_ctx:
                    payload["options"]["num_ctx"] = self.num_ctx
                stats.model_calls += 1
                resp = self.session.post(self.url, json=payload, timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                return (resp.json().get("response") or "").strip()
            except (ConnectionError, ReadTimeout, requests.HTTPError) as exc:
                last_error = exc
                if attempt < MAX_CALL_RETRIES:
                    time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        raise RuntimeError(f"Ollama call failed: {last_error}")

    def generate_json(self, prompt: str, schema_hint: str, stats: CallStats, *, num_predict: Optional[int] = None) -> Optional[Any]:
        raw = self.generate(prompt, stats, num_predict=num_predict)
        parsed = extract_json(raw)
        if parsed is not None:
            return parsed
        repaired = self.generate(
            render_prompt(JSON_REPAIR_PROMPT, schema=schema_hint, bad=raw),
            stats,
            num_predict=800,
        )
        stats.repair_calls += 1
        return extract_json(repaired)


def extract_json(text: str) -> Optional[Any]:
    if not text:
        return None
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```", text, flags=re.S)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except Exception:
            pass

    starts = [pos for pos in (text.find("{"), text.find("[")) if pos != -1]
    if not starts:
        return None
    start = min(starts)
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
                snippet = text[start : i + 1]
                try:
                    return json.loads(snippet)
                except Exception:
                    return None
    return None


# ---------------------------------------------------------------------------
# Light normalization and validation helpers
# ---------------------------------------------------------------------------
def normalize_name(name: str) -> str:
    if not name:
        return ""
    s = strip_accents(normalize_ws(name))
    s = s.strip(" ,.;:[]{}\"'")
    s = re.sub(r"^(?:the\s+)?slave\s+", "", s, flags=re.I)
    s = re.sub(r"^(?:mr|mrs|miss|mst)\.?\s+", "", s, flags=re.I)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = normalize_ws(s)
    tokens: List[str] = []
    for token in s.split():
        low = token.lower()
        if low in {"bin", "bint", "al", "el", "ul", "ibn"}:
            tokens.append("bin" if low == "ibn" else low)
        elif low in {"abu", "umm"}:
            tokens.append(low.title())
        elif low in {"daughter", "son", "of"}:
            tokens.append(low)
        else:
            tokens.append(token[:1].upper() + token[1:])
    return normalize_ws(" ".join(tokens))


def is_valid_name(name: str) -> bool:
    if not name:
        return False
    s = normalize_name(name)
    if len(s) < 2 or re.search(r"\d", s):
        return False
    words = s.split()
    if not words:
        return False
    low_words = {w.lower() for w in words}
    if low_words & NAME_STOPWORDS and len(words) <= 2:
        return False
    if sum(ch.isalpha() for ch in s) < 2:
        return False
    return True


def normalize_place(place: str) -> str:
    if not place:
        return ""
    s = strip_accents(normalize_ws(place))
    s = s.strip(" ,.;:[]{}\"'")
    s = re.sub(r"^\b(?:at|in|to|from|near|via)\b\s+", "", s, flags=re.I)
    s = re.sub(r"\s*\([^)]*\)\s*", " ", s)
    s = normalize_ws(s)
    low = s.lower().replace("-", " ")
    low = re.sub(r"\s+", " ", low)
    mapped = PLACE_MAP.get(low)
    if mapped:
        return mapped
    words = [w for w in s.split() if w]
    if not words:
        return ""
    out: List[str] = []
    for w in words[:6]:
        if w.lower() in {"al", "ul", "el"}:
            out.append(w.lower())
        else:
            out.append(w[:1].upper() + w[1:])
    return normalize_ws(" ".join(out))


def is_valid_place(place: str) -> bool:
    if not place:
        return False
    s = normalize_place(place)
    if not s or re.search(r"\d", s):
        return False
    low = s.lower()
    if low in PLACE_STOPWORDS:
        return False
    if len(s.split()) > 6:
        return False
    if low in {"there", "here", "office", "agency", "residency"}:
        return False
    if re.search(r"\b(h\.m\.s\.?|s\.s\.?|steamship|ship|dhow|vessel|boat)\b", low):
        return False
    return True


def choose_report_type(value: str) -> str:
    value = normalize_ws(value)
    value = LEGACY_REPORT_TYPE_MAP.get(value.lower(), value)
    return value if value in REPORT_TYPES else "correspondence"


def override_report_type_from_ocr(ocr: str, current: str) -> str:
    text = normalize_ws(ocr)
    if STATEMENT_REPORT_PAT.search(text):
        return "statement"
    if TRANSPORT_ADMIN_REPORT_PAT.search(text):
        return "transport/admin"
    return current


def choose_allowed(value: Any, allowed: Iterable[str]) -> str:
    if value is None:
        return ""
    v = normalize_ws(str(value))
    return v if v in set(allowed) else ""


def choose_yes_no_blank(value: Any) -> str:
    v = normalize_ws(str(value or "")).lower()
    if v in {"yes", "no"}:
        return v
    return ""


def extract_doc_year(text: str) -> Optional[int]:
    m = re.search(r"\b(17|18|19|20)\d{2}\b", text or "")
    return int(m.group(0)) if m else None


def parse_day_month(text: str) -> Optional[Tuple[int, int]]:
    s = normalize_ws((text or "").lower().replace(",", " "))
    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([a-z]+)\b", s)
    if not m:
        return None
    day = int(m.group(1))
    mon_name = m.group(2)
    for month_name, month_num in MONTHS.items():
        if mon_name.startswith(month_name[:3]):
            return day, month_num
    return None


def to_iso_date(text: str, doc_year: Optional[int]) -> Tuple[str, str]:
    if not text:
        return "", ""
    s = normalize_ws(text)
    if ISO_DATE_PAT.match(s):
        return s, "explicit"

    m = re.search(r"(?:\bD/?\s*)?(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})\b", s, flags=re.I)
    if m:
        dd, mm, yy = int(m.group(1)), int(m.group(2)), int(m.group(3))
        year = yy
        if yy < 100:
            year = (doc_year // 100) * 100 + yy if doc_year else 1900 + yy
        if 1 <= mm <= 12 and 1 <= dd <= 31:
            return f"{year:04d}-{mm:02d}-{dd:02d}", "explicit" if len(m.group(3)) == 4 else "derived_from_doc"

    m = re.search(r"\b([A-Z][a-z]+)\s+(\d{1,2}),\s*(\d{4})\b", s)
    if m:
        month_name, day, year = m.group(1).lower(), int(m.group(2)), int(m.group(3))
        for k, v in MONTHS.items():
            if month_name.startswith(k[:3]):
                return f"{year:04d}-{v:02d}-{day:02d}", "explicit"

    m = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Z][a-z]+)\s+(\d{4})\b", s)
    if m:
        day, month_name, year = int(m.group(1)), m.group(2).lower(), int(m.group(3))
        for k, v in MONTHS.items():
            if month_name.startswith(k[:3]):
                return f"{year:04d}-{v:02d}-{day:02d}", "explicit"

    dm = parse_day_month(s)
    if dm and doc_year:
        day, month_num = dm
        return f"{doc_year:04d}-{month_num:02d}-{day:02d}", "derived_from_doc"

    return "", ""


def clean_evidence(text: Any) -> str:
    s = normalize_ws(str(text or ""))
    if not s:
        return ""
    words = s.split()
    return " ".join(words[:25])


def render_prompt(template: str, **kwargs: Any) -> str:
    rendered = template
    for key, value in kwargs.items():
        rendered = rendered.replace("{" + key + "}", str(value))
    return rendered


ROLE_NEGATIVE_PATTERNS = [
    r"\bsold\s+(?:me\s+)?to\s+(?:one\s+)?{name}\b",
    r"\bbought\s+by\s+{name}\b",
    r"\bbelonging\s+to\s+{name}\b",
    r"\bowner\s+(?:named\s+)?{name}\b",
    r"\bmaster\s+(?:named\s+)?{name}\b",
    r"\bstatement\s+recorded\s+by\s+{name}\b",
    r"\bletter\s+from\s+{name}\b",
    r"\bsigned\s+before\s+me\s+by\s+{name}\b",
]

ROLE_POSITIVE_PATTERNS = [
    r"\bstatement\s+of\s+(?:slave\s+)?{name}\b",
    r"\bstatement\s+made\s+by\s+{name}\b",
    r"\bslave\s+{name}\b",
    r"\bslave\b.*?\bnamed\s+{name}\b",
    r"\bnamed\s+{name}\b.*?\bslave\b",
    r"\b1\s+slave\b.*?\bnamed\s+{name}\b",
    r"\brefugee\s+slaves?\b.*?\b{name}\b",
    r"\bfor\s+delivery\s+to\s+{name}\b",
    r"\bgrant\b.*?\bcertificate\b.*?\bto\s+{name}\b",
    r"\brecommend\b.*?\bcertificate\b.*?\bfor\s+{name}\b",
    r"\bmanumission\b.*?\b{name}\b",
    r"\bfree\s+status\b.*?\b{name}\b",
    r"\b{name}\b.*?\brequests?\s+repatriation\b",
]

CONFIDENT_ROUTE_PAT = re.compile(
    r"\b(arriv(?:e|ed|ing)|reached|escaped\s+to|took\s+refuge\s+at|went\s+to|came\s+to|brought\s+(?:me\s+)?to|taken\s+to|sent\s+to|forwarded\b|moved\s+to)\b",
    flags=re.I,
)
UNCERTAIN_ROUTE_PAT = re.compile(
    r"\b(request(?:ed)?|desired|wish(?:ed|es)?|intend(?:ed)?|propos(?:ed|es)|recommended|recommendation|delivery|certificate|office|agency|administrative|handling|not\s+clearly\s+completed)\b",
    flags=re.I,
)

TRANSPORT_ADMIN_REPORT_PAT = re.compile(
    r"\b("
    r"repatriation|repatriate|passage|transport|taken\s+to|sent\s+to|for\s+delivery\s+to|"
    r"delivered\s+to|arrange(?:d)?\s+.*?\s+for|maintenance|subsistence|"
    r"provisions?\s+issued|victualled|accommodated\s+on\s+board|provision\s+account|"
    r"certificate\s+delivered|grant\s+certificate|manumission\s+certificate"
    r")\b",
    flags=re.I | re.S,
)
STATEMENT_REPORT_PAT = re.compile(
    r"\b(statement\s+of|statement\s+made\s+by|i\s+was\s+born|i\s+was\s+kidnapped|i\s+request)\b",
    flags=re.I,
)


def normalize_for_match(text: str) -> str:
    text = strip_accents(normalize_ws(text)).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return normalize_ws(text)


def name_compare_tokens(name: str) -> List[str]:
    return [
        tok.lower()
        for tok in normalize_name(name).split()
        if tok.lower() not in {"bin", "bint", "ibn", "son", "daughter", "of", "al", "el", "ul"}
    ]


def names_maybe_same_person(a: str, b: str) -> bool:
    na = normalize_name(a)
    nb = normalize_name(b)
    if not na or not nb:
        return False
    if na.lower() == nb.lower():
        return True
    ta = name_compare_tokens(na)
    tb = name_compare_tokens(nb)
    if not ta or not tb:
        return False
    if ta[0] != tb[0]:
        flat_ratio = difflib.SequenceMatcher(None, "".join(ta), "".join(tb)).ratio()
        return flat_ratio >= 0.92
    seq_ratio = difflib.SequenceMatcher(None, na.lower(), nb.lower()).ratio()
    flat_ratio = difflib.SequenceMatcher(None, "".join(ta), "".join(tb)).ratio()
    overlap = len(set(ta) & set(tb)) / max(len(set(ta)), len(set(tb)), 1)
    if ta == tb:
        return True
    if na.lower() in nb.lower() or nb.lower() in na.lower():
        return True
    return seq_ratio >= 0.9 or flat_ratio >= 0.8 or overlap >= 0.75


def choose_preferred_name(items: List[Dict[str, str]]) -> Dict[str, str]:
    def score(item: Dict[str, str]) -> Tuple[int, int, int, str]:
        name = normalize_name(item.get("name") or "")
        tokens = name_compare_tokens(name)
        return (len(tokens), len(name), len(item.get("evidence") or ""), name.lower())
    return max(items, key=score)


def build_name_regex(name: str) -> Optional[re.Pattern[str]]:
    tokens = [re.escape(tok) for tok in normalize_name(name).split() if tok]
    if not tokens:
        return None
    joined = r"[\s,.;:'\"()\-]+".join(tokens)
    return re.compile(r"\b" + joined + r"\b", flags=re.I)


def iter_name_contexts(name: str, ocr: str, window: int = 140) -> List[str]:
    pattern = build_name_regex(name)
    if not pattern or not ocr:
        return []
    contexts: List[str] = []
    for m in pattern.finditer(ocr):
        start = max(0, m.start() - window)
        end = min(len(ocr), m.end() + window)
        contexts.append(normalize_ws(ocr[start:end]))
    return contexts


def compile_name_phrase(pattern_template: str, name: str) -> re.Pattern[str]:
    tokens = [re.escape(tok) for tok in normalize_name(name).split() if tok]
    joined = r"[\s,.;:'\"()\-]+".join(tokens) if tokens else r""
    return re.compile(pattern_template.format(name=joined), flags=re.I | re.S)


def has_positive_subject_signal(name: str, text: str) -> bool:
    if not text:
        return False
    for template in ROLE_POSITIVE_PATTERNS:
        if compile_name_phrase(template, name).search(text):
            return True
    return False


def has_negative_role_signal(name: str, text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    if re.search(r"\bfree\s*born\b", lower) and re.search(r"\bnot\s+a?\s*slave\b", lower):
        return True
    if re.search(rf"\b(?:major|captain|shaikh|sheikh|secretary|agent|political\s+agent|resident)\b.*?\b{re.escape(normalize_name(name).split()[0].lower())}\b", lower):
        return True
    for template in ROLE_NEGATIVE_PATTERNS:
        if compile_name_phrase(template, name).search(text):
            return True
    return False


def is_freeborn_not_slave_name(name: str, ocr: str) -> bool:
    for ctx in iter_name_contexts(name, ocr):
        low = ctx.lower()
        if "free born" in low and ("not a slave" in low or "not slave" in low):
            return True
    return False


def keep_subject_name(name: str, evidence: str, ocr: str) -> bool:
    if not is_valid_name(name):
        return False
    if is_freeborn_not_slave_name(name, ocr):
        return False
    texts = [clean_evidence(evidence)] + iter_name_contexts(name, ocr)
    pos = sum(1 for txt in texts if has_positive_subject_signal(name, txt))
    neg = sum(1 for txt in texts if has_negative_role_signal(name, txt))
    if pos > 0:
        return True
    if neg > 0:
        return False
    # conservative fallback: only keep on very strong remaining local wording
    joined = " ".join(texts).lower()
    strong_local = [
        "slave named",
        "refugee slave",
        "fugitive slave",
        "statement of slave",
        "statement made by",
        "grant certificate to",
        "recommend certificate for",
        "requests repatriation",
    ]
    return any(phrase in joined for phrase in strong_local)


def filter_named_people(named_people: List[Dict[str, str]], ocr: str) -> List[Dict[str, str]]:
    kept: List[Dict[str, str]] = []
    for item in named_people:
        if keep_subject_name(item.get("name", ""), item.get("evidence", ""), ocr):
            kept.append(item)
    return merge_named_people(kept)


def parse_first_date_in_text(text: str, doc_year: Optional[int]) -> Tuple[str, str, str]:
    if not text:
        return "", "", ""
    s = normalize_ws(text)
    m = re.search(r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+(?:\s+\d{4})?)", s)
    if m:
        iso, conf = to_iso_date(m.group(1), doc_year)
        return iso, conf, m.group(1)
    iso, conf = to_iso_date(s, doc_year)
    return iso, conf, s if iso else ""


def first_text_position(snippet: str, ocr: str) -> int:
    if not snippet or not ocr:
        return 10**9
    idx = ocr.lower().find(snippet.lower())
    if idx != -1:
        return idx
    norm_snippet = normalize_for_match(snippet)
    norm_ocr = normalize_for_match(ocr)
    idx = norm_ocr.find(norm_snippet)
    return idx if idx != -1 else 10**9


def first_place_position(place: str, evidence: str, ocr: str) -> int:
    pos = first_text_position(evidence, ocr)
    if pos != 10**9:
        return pos
    pattern = build_name_regex(place)
    if pattern:
        m = pattern.search(ocr)
        if m:
            return m.start()
    return first_text_position(place, ocr)


def is_uncertain_place_text(text: str) -> bool:
    return bool(UNCERTAIN_ROUTE_PAT.search(text or "")) and not bool(re.search(r"\b(arriv(?:ed|ing)|reached|escaped\s+to|went\s+to|came\s+to)\b", text or "", flags=re.I))


def is_confident_place_text(text: str) -> bool:
    return bool(CONFIDENT_ROUTE_PAT.search(text or ""))


def infer_forwarding_transport_rows(name: str, ocr: str, page: int, doc_year: Optional[int]) -> List[Dict[str, Any]]:
    lower = ocr.lower()
    if normalize_name(name).lower() not in lower.lower() or not re.search(r"\b(forwarded|forwarding|arriving|arrived)\b", lower):
        return []
    rows: List[Dict[str, Any]] = []
    src_match = re.search(r"^\s*from\s*-\s*.*?,\s*([A-Za-z][A-Za-z' -]+?)\.\s*$", ocr, flags=re.I | re.M)
    if not src_match:
        src_match = re.search(r"from\s*-\s*(?:.*?,\s*)?([A-Za-z][A-Za-z' -]+?)\.\s*$", ocr, flags=re.I | re.M)
    dst_match = re.search(r"arriv(?:ing|ed)\s+([A-Za-z][A-Za-z' -]*[A-Za-z])(?=\s+(?:about\s+the|on|about)\b|[.,;\n])(?:\s+about\s+the\s+|\s+on\s+|\s+about\s+)?([^.,;\n]+)?", ocr, flags=re.I)
    src = normalize_place(src_match.group(1)) if src_match else ""
    dst = normalize_place(dst_match.group(1)) if dst_match else ""
    if src and is_valid_place(src):
        rows.append({
            "Name": name, "Page": page, "Place": src, "Order": 1,
            "Arrival Date": "", "Date Confidence": "", "Time Info": "",
            "_evidence": clean_evidence(src_match.group(0) if src_match else src),
            "_force_rank": 1,
        })
    if dst and is_valid_place(dst):
        arrival_date = ""
        date_conf = ""
        time_text = ""
        if dst_match and dst_match.group(2):
            arrival_date, date_conf, time_text = parse_first_date_in_text(dst_match.group(2), doc_year)
            if not arrival_date:
                time_text = normalize_ws(dst_match.group(2))
        rows.append({
            "Name": name, "Page": page, "Place": dst, "Order": 2 if src else 1,
            "Arrival Date": arrival_date, "Date Confidence": date_conf, "Time Info": time_text,
            "_evidence": clean_evidence(dst_match.group(0) if dst_match else dst),
            "_force_rank": 2 if src else 1,
        })
    return rows


def reconcile_place_rows(rows: List[Dict[str, Any]], ocr: str, name: str, page: int, doc_year: Optional[int]) -> List[Dict[str, Any]]:
    work = [dict(row) for row in rows]
    work.extend(infer_forwarding_transport_rows(name, ocr, page, doc_year))
    work = dedupe_place_rows(work, drop_internal=False)
    if not work:
        return []

    for row in work:
        text = normalize_ws(f"{row.get('_evidence', '')} {row.get('Time Info', '')}")
        row["_position"] = first_place_position(str(row.get("Place") or ""), text, ocr)
        order = int(row.get("Order", 0) or 0)
        if order <= 0 and row.get("Arrival Date") and not is_uncertain_place_text(text):
            row["_promote"] = True
        elif order <= 0 and is_confident_place_text(text) and not is_uncertain_place_text(text):
            row["_promote"] = True
        else:
            row["_promote"] = False

    route_rows: List[Dict[str, Any]] = []
    zero_rows: List[Dict[str, Any]] = []
    for row in work:
        order = int(row.get("Order", 0) or 0)
        if order > 0 or row.get("_promote") or row.get("_force_rank"):
            route_rows.append(row)
        else:
            zero_rows.append(row)

    route_rows.sort(key=lambda r: (
        int(r.get("_force_rank", 10**6) or 10**6),
        int(r.get("_position", 10**9) or 10**9),
        int(r.get("Order", 10**6) or 10**6),
        str(r.get("Place") or "").lower(),
    ))
    for idx, row in enumerate(route_rows, start=1):
        row["Order"] = idx

    zero_rows.sort(key=lambda r: (int(r.get("_position", 10**9) or 10**9), str(r.get("Place") or "").lower()))
    for row in zero_rows:
        row["Order"] = 0

    return dedupe_place_rows(route_rows + zero_rows)


# ---------------------------------------------------------------------------
# Parsing model outputs
# ---------------------------------------------------------------------------
def parse_page_decision(obj: Any) -> PageDecision:
    if not isinstance(obj, dict):
        return PageDecision(True, None, "correspondence", "")
    should_extract = bool(obj.get("should_extract", True))
    skip_reason = obj.get("skip_reason")
    skip_reason = normalize_ws(str(skip_reason)) if skip_reason not in (None, "null") else ""
    if skip_reason not in {"", "index", "record_metadata", "bad_ocr"}:
        skip_reason = ""
    report_type = choose_report_type(str(obj.get("report_type") or "correspondence"))
    evidence = clean_evidence(obj.get("evidence"))
    if skip_reason:
        return PageDecision(False, skip_reason, report_type, evidence)
    return PageDecision(should_extract, None, report_type, evidence)


def parse_named_people(obj: Any) -> List[Dict[str, str]]:
    if not isinstance(obj, dict):
        return []
    merged: Dict[str, Dict[str, str]] = {}
    for item in obj.get("named_people") or []:
        if not isinstance(item, dict):
            continue
        name = normalize_name(str(item.get("name") or ""))
        if not is_valid_name(name):
            continue
        key = name.lower()
        evidence = clean_evidence(item.get("evidence"))
        current = merged.get(key)
        if current is None or len(name) > len(current["name"]) or len(evidence) > len(current["evidence"]):
            merged[key] = {"name": name, "evidence": evidence}
    return list(merged.values())


def merge_named_people(*groups: List[Dict[str, str]]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    for group in groups:
        items.extend(group or [])
    clusters: List[List[Dict[str, str]]] = []
    for item in items:
        placed = False
        for cluster in clusters:
            if any(names_maybe_same_person(item["name"], other["name"]) for other in cluster):
                cluster.append(item)
                placed = True
                break
        if not placed:
            clusters.append([item])
    merged = [choose_preferred_name(cluster) for cluster in clusters if cluster]
    return sorted(merged, key=lambda x: x["name"].lower())


def parse_meta(obj: Any, name: str, page: int, report_type: str) -> Dict[str, Any]:
    data = {
        "Name": name,
        "Page": page,
        "Report Type": report_type,
        "Crime Type": "",
        "Whether abuse": "",
        "Conflict Type": "",
        "Trial": "",
        "Amount paid": "",
    }
    if not isinstance(obj, dict):
        return data
    data["Crime Type"] = choose_allowed(obj.get("crime_type"), CRIME_TYPES)
    data["Whether abuse"] = choose_yes_no_blank(obj.get("whether_abuse"))
    data["Conflict Type"] = choose_allowed(obj.get("conflict_type"), CONFLICT_TYPES)
    data["Trial"] = choose_allowed(obj.get("trial"), TRIAL_TYPES)
    amount = normalize_ws(str(obj.get("amount_paid") or ""))
    data["Amount paid"] = "" if amount.lower() in {"null", "none"} else amount
    return data


def parse_places(obj: Any, name: str, page: int, doc_year: Optional[int]) -> List[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return []
    rows: List[Dict[str, Any]] = []
    for item in obj.get("places") or []:
        if not isinstance(item, dict):
            continue
        place = normalize_place(str(item.get("place") or ""))
        if not is_valid_place(place):
            continue
        try:
            order = int(item.get("order", 0) or 0)
            order = max(order, 0)
        except Exception:
            order = 0

        raw_date = "" if item.get("arrival_date") in (None, "null") else normalize_ws(str(item.get("arrival_date") or ""))
        arrival_date, inferred_conf = to_iso_date(raw_date, doc_year)
        date_confidence = normalize_ws(str(item.get("date_confidence") or ""))
        if date_confidence not in DATE_CONFIDENCE:
            date_confidence = inferred_conf
        if not arrival_date:
            date_confidence = ""

        time_text = normalize_ws(str(item.get("time_text") or ""))
        if raw_date and not arrival_date and raw_date.lower() not in time_text.lower():
            time_text = normalize_ws(f"{raw_date}; {time_text}" if time_text else raw_date)

        evidence = clean_evidence(item.get("evidence"))
        rows.append({
            "Name": name,
            "Page": page,
            "Place": place,
            "Order": order,
            "Arrival Date": arrival_date,
            "Date Confidence": date_confidence,
            "Time Info": time_text,
            "_evidence": evidence,
        })
    return dedupe_place_rows(rows, drop_internal=False)


def model_filter_named_people(client: "OllamaClient", ocr: str, candidates: List[Dict[str, str]], stats: CallStats, *, stage: str) -> List[Dict[str, str]]:
    if not candidates:
        return []
    schema = '{"named_people":[{"name":"...","evidence":"..."}]}'
    payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    obj = client.generate_json(
        render_prompt(NAME_FILTER_PROMPT, stage=stage, candidate_names_json=payload, ocr=ocr),
        schema,
        stats,
        num_predict=900,
    )
    filtered = parse_named_people(obj)
    allowed = {normalize_name(c.get("name") or "").lower(): c for c in candidates}
    out: List[Dict[str, str]] = []
    seen = set()
    for item in filtered:
        key = normalize_name(item.get("name") or "").lower()
        if not key or key not in allowed or key in seen:
            continue
        ev = clean_evidence(item.get("evidence") or allowed[key].get("evidence") or "")
        out.append({"name": allowed[key]["name"], "evidence": ev})
        seen.add(key)
    return out


def model_verify_named_people(client: "OllamaClient", ocr: str, candidates: List[Dict[str, str]], stats: CallStats) -> List[Dict[str, str]]:
    if not candidates:
        return []
    schema = '{"named_people":[{"name":"...","evidence":"..."}]}'
    payload = json.dumps(candidates, ensure_ascii=False, indent=2)
    obj = client.generate_json(
        render_prompt(NAME_VERIFY_PROMPT, candidate_names_json=payload, ocr=ocr),
        schema,
        stats,
        num_predict=900,
    )
    filtered = parse_named_people(obj)
    allowed = {normalize_name(c.get("name") or "").lower(): c for c in candidates}
    out: List[Dict[str, str]] = []
    seen = set()
    for item in filtered:
        key = normalize_name(item.get("name") or "").lower()
        if not key or key not in allowed or key in seen:
            continue
        ev = clean_evidence(item.get("evidence") or allowed[key].get("evidence") or "")
        out.append({"name": allowed[key]["name"], "evidence": ev})
        seen.add(key)
    return out

def place_row_score(row: Dict[str, Any]) -> Tuple[int, int, int, int, int]:
    conf_rank = {"": 0, "unknown": 1, "derived_from_doc": 2, "explicit": 3}
    return (
        1 if int(row.get("Order", 0) or 0) > 0 else 0,
        1 if row.get("Arrival Date") else 0,
        conf_rank.get(str(row.get("Date Confidence") or ""), 0),
        1 if row.get("Time Info") else 0,
        len(str(row.get("_evidence") or "")),
    )


def verify_place_rows_need_retry(rows: List[Dict[str, Any]]) -> Optional[str]:
    if not rows:
        return "Verifier returned no places."
    positive = [r for r in rows if int(r.get("Order", 0) or 0) > 0]
    orders = [int(r.get("Order", 0) or 0) for r in positive]
    if orders != list(range(1, len(orders) + 1)):
        return "Positive orders must be consecutive 1..n."
    seen = set()
    for row in rows:
        place = str(row.get("Place") or "").strip().lower()
        if place in seen:
            return "Duplicate final places remain after verification."
        seen.add(place)
        if not row.get("Arrival Date") and row.get("Date Confidence"):
            return "Date confidence must be blank when arrival_date is blank."
    dated_positive = [r for r in positive if r.get("Arrival Date")]
    for a, b in zip(dated_positive, dated_positive[1:]):
        if str(a.get("Arrival Date")) > str(b.get("Arrival Date")):
            return "Positive route order conflicts with arrival dates."
    return None


def dedupe_place_rows(rows: List[Dict[str, Any]], *, drop_internal: bool = True) -> List[Dict[str, Any]]:
    if not rows:
        return []
    best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("Name") or "").lower(), str(row.get("Place") or "").lower())
        current = best.get(key)
        candidate = dict(row)
        if current is None or place_row_score(candidate) > place_row_score(current):
            merged = dict(candidate)
            if current is not None:
                if not merged.get("Arrival Date") and current.get("Arrival Date"):
                    merged["Arrival Date"] = current["Arrival Date"]
                    merged["Date Confidence"] = current.get("Date Confidence", "")
                if not merged.get("Time Info") and current.get("Time Info"):
                    merged["Time Info"] = current["Time Info"]
                if not merged.get("_evidence") and current.get("_evidence"):
                    merged["_evidence"] = current["_evidence"]
            best[key] = merged
        else:
            if not current.get("Arrival Date") and candidate.get("Arrival Date"):
                current["Arrival Date"] = candidate["Arrival Date"]
                current["Date Confidence"] = candidate.get("Date Confidence", "")
            if not current.get("Time Info") and candidate.get("Time Info"):
                current["Time Info"] = candidate["Time Info"]
            if not current.get("_evidence") and candidate.get("_evidence"):
                current["_evidence"] = candidate["_evidence"]

    positives = [r for r in best.values() if int(r.get("Order", 0) or 0) > 0]
    zeroes = [r for r in best.values() if int(r.get("Order", 0) or 0) == 0]
    positives.sort(key=lambda r: (int(r.get("Order", 0) or 0), str(r.get("Arrival Date") or ""), r["Place"].lower()))
    for idx, row in enumerate(positives, start=1):
        row["Order"] = idx
    zeroes.sort(key=lambda r: (str(r.get("Arrival Date") or ""), r["Place"].lower()))
    out = positives + zeroes
    if drop_internal:
        for row in out:
            row.pop("_evidence", None)
            row.pop("_position", None)
            row.pop("_promote", None)
            row.pop("_force_rank", None)
    return out


def merge_place_date_enrichment(base_rows: List[Dict[str, Any]], enriched_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not base_rows or not enriched_rows:
        return dedupe_place_rows(base_rows, drop_internal=False)
    by_place = {row["Place"].lower(): row for row in base_rows}
    for row in enriched_rows:
        target = by_place.get(row["Place"].lower())
        if not target:
            continue
        if not target.get("Arrival Date") and row.get("Arrival Date"):
            target["Arrival Date"] = row["Arrival Date"]
            target["Date Confidence"] = row["Date Confidence"]
        if len(str(row.get("Time Info") or "")) > len(str(target.get("Time Info") or "")):
            target["Time Info"] = row["Time Info"]
    return dedupe_place_rows(list(by_place.values()), drop_internal=False)


def blank_place_row(name: str, page: int) -> Dict[str, Any]:
    return {
        "Name": name,
        "Page": page,
        "Place": "",
        "Order": "",
        "Arrival Date": "",
        "Date Confidence": "",
        "Time Info": "",
    }


# ---------------------------------------------------------------------------
# Model-first extraction helpers
# ---------------------------------------------------------------------------
def model_page_decision(client: "OllamaClient", ocr: str, stats: CallStats, *, report_type_override: Optional[str]) -> PageDecision:
    if report_type_override:
        return PageDecision(True, None, choose_report_type(report_type_override), "override")
    schema = '{"should_extract":true,"skip_reason":null,"report_type":"statement","evidence":"..."}'
    obj = client.generate_json(render_prompt(PAGE_CLASSIFY_PROMPT, ocr=ocr), schema, stats, num_predict=500)
    decision = parse_page_decision(obj)
    decision.report_type = override_report_type_from_ocr(ocr, decision.report_type)
    return decision


def model_named_people(client: "OllamaClient", ocr: str, stats: CallStats) -> List[Dict[str, str]]:
    schema = '{"named_people":[{"name":"...","evidence":"..."}]}'
    first_raw = parse_named_people(client.generate_json(render_prompt(NAME_PASS_PROMPT, ocr=ocr), schema, stats, num_predict=900))
    first = model_filter_named_people(client, ocr, first_raw, stats, stage="pass-1") or first_raw
    second_raw = parse_named_people(client.generate_json(render_prompt(NAME_RECALL_PROMPT, ocr=ocr), schema, stats, num_predict=900))
    second = model_filter_named_people(client, ocr, second_raw, stats, stage="recall-2") or second_raw
    merged = merge_named_people(first, second)
    verified = model_verify_named_people(client, ocr, merged, stats) or merged
    return filter_named_people(verified, ocr)


def model_meta_for_name(client: "OllamaClient", ocr: str, name: str, page: int, report_type: str, stats: CallStats) -> Dict[str, Any]:
    schema = (
        '{"name":"%s","page":%d,"report_type":"%s","crime_type":null,'
        '"whether_abuse":"","conflict_type":null,"trial":null,"amount_paid":null,'
        '"evidence":{"crime_type":null,"whether_abuse":null,"conflict_type":null,"trial":null,"amount_paid":null}}'
    ) % (name, page, report_type)
    obj = client.generate_json(
        render_prompt(META_PASS_PROMPT, name=name, page=page, report_type=report_type, ocr=ocr),
        schema,
        stats,
        num_predict=1000,
    )
    return parse_meta(obj, name, page, report_type)


def model_places_for_name(client: "OllamaClient", ocr: str, name: str, page: int, stats: CallStats) -> List[Dict[str, Any]]:
    doc_year = extract_doc_year(ocr)
    candidate_schema = '{"name":"%s","places":[{"place":"...","time_text":null,"evidence":"..."}]}' % name
    final_schema = '{"name":"%s","places":[{"place":"...","order":0,"arrival_date":null,"date_confidence":"unknown","time_text":null,"evidence":"..."}]}' % name

    candidates = parse_places(
        client.generate_json(render_prompt(PLACE_PASS_PROMPT, name=name, ocr=ocr), candidate_schema, stats, num_predict=1000),
        name,
        page,
        doc_year,
    )
    candidates = dedupe_place_rows(candidates, drop_internal=False)
    if not candidates:
        return []

    candidate_payload = json.dumps([
        {
            "place": row["Place"],
            "time_text": row["Time Info"] or None,
            "evidence": row.get("_evidence") or None,
        }
        for row in candidates
    ], ensure_ascii=False, indent=2)

    issues = ""
    final_rows: List[Dict[str, Any]] = []
    for attempt in range(2):
        prompt = render_prompt(
            PLACE_VERIFY_PROMPT,
            name=name,
            page=page,
            candidate_places_json=(candidate_payload if not issues else candidate_payload + "\n\nIssues to fix:\n- " + issues),
            ocr=ocr,
        )
        verified_obj = client.generate_json(prompt, final_schema, stats, num_predict=1200)
        final_rows = parse_places(verified_obj, name, page, doc_year)
        final_rows = dedupe_place_rows(final_rows, drop_internal=False)
        issue = verify_place_rows_need_retry(final_rows)
        if not issue:
            return dedupe_place_rows(final_rows)
        issues = issue

    # Safe fallback: preserve validated candidate mentions as order-0 rows
    # rather than dropping all place information when final adjudication fails.
    if final_rows:
        return dedupe_place_rows(final_rows)
    return dedupe_place_rows(candidates)


# ---------------------------------------------------------------------------
# Page pipeline
# ---------------------------------------------------------------------------
def process_page(
    client: "OllamaClient",
    path: pathlib.Path,
    report_type_override: Optional[str],
    stats: CallStats,
) -> Tuple[str, str, List[Dict[str, Any]], List[Dict[str, Any]], str]:
    page = page_number_from_path(path)
    ocr = clean_ocr(path.read_text(encoding="utf-8", errors="ignore"))
    if not normalize_ws(ocr):
        return "skip:bad_ocr", "correspondence", [], [], "empty OCR"

    decision = model_page_decision(client, ocr, stats, report_type_override=report_type_override)
    if not decision.should_extract:
        skip_reason = decision.skip_reason or "bad_ocr"
        status = f"skip:{skip_reason}"
        if status not in STATUS_VALUES:
            status = "skip:bad_ocr"
        return status, decision.report_type, [], [], decision.evidence or skip_reason

    names = model_named_people(client, ocr, stats)
    if not names:
        return "no_named_people", decision.report_type, [], [], "no valid named subjects"

    detail_rows: List[Dict[str, Any]] = []
    place_rows: List[Dict[str, Any]] = []

    for item in names:
        name = item["name"]
        meta = model_meta_for_name(client, ocr, name, page, decision.report_type, stats)
        detail_rows.append(meta)

        places = model_places_for_name(client, ocr, name, page, stats)
        if places:
            place_rows.extend(places)
        else:
            place_rows.append(blank_place_row(name, page))

    return "ok", decision.report_type, detail_rows, place_rows, f"{len(names)} named people"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Model-first OCR extractor for historical slavery/manumission pages.")
    ap.add_argument("--in_dir", default="/data/input", help="Directory containing page-level OCR .txt files")
    ap.add_argument("--out_dir", default="/data/output", help="Directory for CSV outputs")
    ap.add_argument("--text_out_dir", default="/data/text_out")  # compatibility only
    ap.add_argument("--log_dir", default="/data/logs", help="Directory for run.log")
    ap.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama /api/generate URL")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name")
    ap.add_argument("--num-predict", type=int, default=DEFAULT_NUM_PREDICT, help="Default Ollama num_predict")
    ap.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="Default Ollama num_ctx")
    ap.add_argument("--report-type", default=None, help="Optional fixed report type override for all pages")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging to run.log")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    log_dir = pathlib.Path(args.log_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_dir, args.verbose)

    detail_path = out_dir / "Detailed info.csv"
    place_path = out_dir / "name place.csv"
    status_path = out_dir / "run_status.csv"

    files = sorted(pathlib.Path(args.in_dir).glob("*.txt"), key=lambda p: (page_number_from_path(p), p.name))
    if not files:
        logger.error("No .txt files found in %s", args.in_dir)
        progress(f"[0/0] no .txt files found in {args.in_dir}")
        write_csv(detail_path, [], DETAIL_COLUMNS)
        write_csv(place_path, [], PLACE_COLUMNS)
        write_csv(status_path, [], STATUS_COLUMNS)
        return

    client = OllamaClient(
        url=args.ollama_url,
        model=args.model,
        num_predict=args.num_predict,
        num_ctx=args.num_ctx,
    )

    all_detail_rows: List[Dict[str, Any]] = []
    all_place_rows: List[Dict[str, Any]] = []
    status_rows: List[Dict[str, Any]] = []

    write_csv(detail_path, all_detail_rows, DETAIL_COLUMNS)
    write_csv(place_path, all_place_rows, PLACE_COLUMNS)
    write_csv(status_path, status_rows, STATUS_COLUMNS)

    progress(f"Starting extraction: {len(files)} page(s) | model={args.model} | input={args.in_dir}")

    total = len(files)
    for idx, path in enumerate(files, start=1):
        t0 = time.time()
        stats = CallStats()
        status = "error"
        note = ""
        report_type = choose_report_type(args.report_type or "correspondence")
        detail_rows: List[Dict[str, Any]] = []
        place_rows: List[Dict[str, Any]] = []
        page = page_number_from_path(path)

        progress(f"[{idx}/{total}] page={page} file={path.name} starting")

        try:
            status, report_type, detail_rows, place_rows, note = process_page(
                client=client,
                path=path,
                report_type_override=args.report_type,
                stats=stats,
            )
            if status == "ok":
                all_detail_rows.extend(detail_rows)
                all_place_rows.extend(place_rows)
        except Exception as exc:
            logger.exception("Failed page %s", path.name)
            status = "error"
            note = str(exc)

        elapsed = round(time.time() - t0, 2)
        status_rows.append({
            "page": page,
            "filename": path.name,
            "status": status,
            "named_people": len(detail_rows),
            "detail_rows": len(detail_rows),
            "place_rows": len(place_rows),
            "model_calls": stats.model_calls,
            "repair_calls": stats.repair_calls,
            "elapsed_seconds": elapsed,
            "note": note,
        })

        write_csv(detail_path, all_detail_rows, DETAIL_COLUMNS)
        write_csv(place_path, all_place_rows, PLACE_COLUMNS)
        write_csv(status_path, status_rows, STATUS_COLUMNS)

        progress(
            f"[{idx}/{total}] done page={page} status={status} report={report_type} "
            f"people={len(detail_rows)} places={len(place_rows)} "
            f"calls={stats.model_calls}/{stats.repair_calls} time={elapsed}s"
        )

        logger.info(
            "page=%s status=%s report_type=%s named_people=%s place_rows=%s model_calls=%s repair_calls=%s",
            page,
            status,
            report_type,
            len(detail_rows),
            len(place_rows),
            stats.model_calls,
            stats.repair_calls,
        )

    progress(f"Finished: {len(status_rows)} page(s) processed | detail_rows={len(all_detail_rows)} | place_rows={len(all_place_rows)}")


if __name__ == "__main__":
    main()
