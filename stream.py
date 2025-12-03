"""
Streamlit app: Reference → RIS/BibTeX/CSV
Features (refactored):
- Paste or upload PDF references
- Parse references using OpenAI (required)
- Search Crossref / PubMed / EuropePMC / SemanticScholar / OpenAlex by extracted title
- Score matches and pick best source
- Show expandable comparison panels (Found vs AI-parsed)
- Let user choose per-reference which metadata to export
- Export RIS / BibTeX / CSV

Requirements: streamlit, openai>=1.0.0, pypdf, requests
"""

import os
import re
import json
import time
import csv
import io
import hashlib
import requests
import streamlit as st
from typing import List, Tuple, Optional, Dict, Any
from difflib import SequenceMatcher
from pypdf import PdfReader
from openai import OpenAI

st.set_page_config(page_title="Reference → RIS (compare Found vs AI)", layout="wide")

# -------------------------
# Config / secrets
# -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_THRESHOLD = 0.3

if not OPENAI_API_KEY:
    st.warning("OpenAI API key not found — set OPENAI_API_KEY in Streamlit secrets or environment.")

# Create OpenAI client (if key present)
client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)

# In-run cache
API_CACHE: Dict[str, Any] = {}

def cached(key: str, fn, *args, **kwargs):
    if key in API_CACHE:
        return API_CACHE[key]
    res = fn(*args, **kwargs)
    API_CACHE[key] = res
    return res

# -------------------------
# Utilities
# -------------------------

def normalize_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()


def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

def text_hash_for_dedupe(s: str) -> str:
    s2 = normalize_text((s or "").lower())
    s2 = re.sub(r"\b(19|20)\d{2}\b", "", s2)
    s2 = re.sub(r"\W+", " ", s2)
    return hashlib.sha1(s2.encode("utf-8")).hexdigest()

# -------------------------
# OpenAI parsing (required)
# -------------------------

def openai_parse_reference(ref_text: str, model: str = OPENAI_MODEL, max_tokens: int = 600) -> Optional[Dict[str, Any]]:
    """Parse a single bibliographic reference into structured JSON using OpenAI.
    Returns a dict with fields: authors (list), title, journal, year, volume, issue, pages, doi
    """
    if client is None:
        st.error("OpenAI client not configured. Please set OPENAI_API_KEY.")
        return None

    prompt = (
        "You are a precise metadata extractor. Convert the following single bibliographic reference into a JSON object with these exact fields:\n"
        "- authors: array of strings (each 'Family, Given' or 'Given Family')\n"
        "- title: string (REQUIRED - extract the main publication title)\n"
        "- journal: string\n"
        "- year: string (4-digit) or empty\n"
        "- volume: string or empty\n"
        "- issue: string or empty\n"
        "- pages: string or empty\n"
        "- doi: string or empty\n\n"
        "Return ONLY valid JSON and nothing else. If a field is unavailable, use an empty string or empty array.\n\n"
        f"Reference:\n\"\"\"\n{ref_text}\n\"\"\""
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        content = resp.choices[0].message["content"]
        # Try to extract JSON (robust to stray text)
        jtext = content.strip()
        # If assistant wraps with ``` or backticks, remove them
        jtext = re.sub(r"^```\w*|```$", "", jtext).strip()
        parsed = json.loads(jtext)

        return {
            "authors": parsed.get("authors", []),
            "title": parsed.get("title", "").strip(),
            "journal": parsed.get("journal", "").strip(),
            "year": str(parsed.get("year", "")).strip(),
            "volume": parsed.get("volume", "").strip(),
            "issue": parsed.get("issue", "").strip(),
            "pages": parsed.get("pages", "").strip(),
            "doi": parsed.get("doi", "").strip(),
        }
    except Exception as e:
        st.error(f"OpenAI parsing error: {e}")
        return None


def extract_text_from_pdf(file_obj) -> str:
    """Extract text from PDF file using pypdf."""
    try:
        reader = PdfReader(file_obj)
        text_parts = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                text_parts.append(txt)
        return "\n".join(text_parts)
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {e}"

# -------------------------
# Multi-source search by title
# -------------------------
# Each function returns (metadata_dict | None, error_message | None)

def crossref_search_title(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not title:
        return None, None
    key = "cr:" + title[:240]

    def _fn(t):
        try:
            params = {"query.title": t, "rows": 1}
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("message", {}).get("items", [])
            if not items:
                return None, None
            item = items[0]
            auths = []
            for a in item.get("author", []):
                auths.append({"family": a.get("family", ""), "given": a.get("given", "")})
            return ({
                "title": item.get("title", [""])[0] if item.get("title") else "",
                "authors": auths,
                "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
                "year": str(item.get("issued", {}).get("date-parts", [[""]])[0][0]) if item.get("issued") else "",
                "volume": str(item.get("volume", "")),
                "issue": str(item.get("issue", "")),
                "pages": item.get("page", ""),
                "doi": item.get("DOI", ""),
            }, None)
        except Exception as e:
            return None, str(e)

    return cached(key, _fn, title)


def pubmed_search_title(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not title:
        return None, None
    key = "pm:" + title[:240]

    def _fn(t):
        try:
            params = {"db": "pubmed", "term": t, "retmode": "xml", "retmax": 1}
            r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=10)
            r.raise_for_status()
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.content)
            ids = root.findall(".//Id")
            if not ids:
                return None, None
            pmid = ids[0].text
            params2 = {"db": "pubmed", "id": pmid, "retmode": "xml"}
            r2 = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=params2, timeout=10)
            r2.raise_for_status()
            root2 = ET.fromstring(r2.content)
            title_elem = root2.find(".//ArticleTitle")
            title_text = title_elem.text if title_elem is not None else ""
            auths = []
            for author in root2.findall(".//Author"):
                ln = author.find("LastName")
                fn = author.find("ForeName")
                if ln is not None:
                    auths.append({"family": ln.text or "", "given": fn.text or ""})
            journal_elem = root2.find(".//Journal/Title")
            journal_text = journal_elem.text if journal_elem is not None else ""
            year_elem = root2.find(".//PubDate/Year")
            year_text = year_elem.text if year_elem is not None else ""
            volume_elem = root2.find(".//Volume")
            volume_text = volume_elem.text if volume_elem is not None else ""
            issue_elem = root2.find(".//Issue")
            issue_text = issue_elem.text if issue_elem is not None else ""
            pages_elem = root2.find(".//MedlinePgn")
            pages_text = pages_elem.text if pages_elem is not None else ""
            doi = ""
            # PubMed XML may contain ArticleIdList with DOI
            for aid in root2.findall('.//ArticleId'):
                if aid.get('IdType') == 'doi':
                    doi = aid.text or ''
            return ({
                "title": title_text,
                "authors": auths,
                "journal": journal_text,
                "year": year_text,
                "volume": volume_text,
                "issue": issue_text,
                "pages": pages_text,
                "doi": doi,
            }, None)
        except Exception as e:
            return None, str(e)

    return cached(key, _fn, title)


def europepmc_search_title(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not title:
        return None, None
    key = "epmc:" + title[:240]

    def _fn(t):
        try:
            params = {"query": t, "format": "json", "pageSize": 1}
            r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            results = data.get("resultList", {}).get("result", [])
            if not results:
                return None, None
            item = results[0]
            auths = []
            for a in item.get("authorList", {}).get("author", []):
                auths.append({"family": a.get("lastName", ""), "given": a.get("firstName", "")})
            return ({
                "title": item.get("title", ""),
                "authors": auths,
                "journal": item.get("journalTitle", ""),
                "year": str(item.get("pubYear", "")),
                "volume": str(item.get("journalVolume", "")),
                "issue": str(item.get("issue", "")),
                "pages": item.get("pageInfo", ""),
                "doi": item.get("doi", ""),
            }, None)
        except Exception as e:
            return None, str(e)

    return cached(key, _fn, title)


def semanticscholar_search_title(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not title:
        return None, None
    key = "ss:" + title[:240]

    def _fn(t):
        try:
            params = {"query": t, "limit": 1}
            r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            papers = data.get("data", [])
            if not papers:
                return None, None
            item = papers[0]
            auths = [{"family": a.get("name", ""), "given": ""} for a in item.get("authors", [])]
            return ({
                "title": item.get("title", ""),
                "authors": auths,
                "journal": item.get("venue", ""),
                "year": str(item.get("year", "")),
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": item.get("externalIds", {}).get("DOI", ""),
            }, None)
        except Exception as e:
            return None, str(e)

    return cached(key, _fn, title)


def openalex_search_title(title: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not title:
        return None, None
    key = "oa:" + title[:240]

    def _fn(t):
        try:
            params = {"search": t}
            headers = {"Accept": "application/json"}
            r = requests.get("https://api.openalex.org/works", params=params, headers=headers, timeout=10)
            r.raise_for_status()
            data = r.json()
            results = data.get("results", [])
            if not results:
                return None, None
            item = results[0]
            auths = []
            for a in item.get("authorships", []):
                author = a.get("author", {})
                auths.append({"family": author.get("display_name", ""), "given": ""})
            return ({
                "title": item.get("title", ""),
                "authors": auths,
                "journal": item.get("primary_location", {}).get("source", {}).get("display_name", ""),
                "year": str(item.get("publication_year", "")),
                "volume": item.get("biblio", {}).get("volume", ""),
                "issue": item.get("biblio", {}).get("issue", ""),
                "pages": item.get("biblio", {}).get("first_page", ""),
                "doi": (item.get("doi", "") or "").replace("https://doi.org/", ""),
            }, None)
        except Exception as e:
            return None, str(e)

    return cached(key, _fn, title)

# -------------------------
# Scoring & compare
# -------------------------

def first_family_from_list(authors_list: List[str]) -> str:
    if not authors_list:
        return ""
    first = authors_list[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    toks = first.split()
    return toks[-1] if toks else ""


def family_list_from_parsed(authors_list: List[str]) -> List[str]:
    fams = []
    for a in authors_list:
        if isinstance(a, str):
            a = a.strip()
            if "," in a:
                fams.append(a.split(",")[0].strip())
            else:
                toks = a.split()
                if toks:
                    fams.append(toks[-1])
    return fams


def family_list_from_found(found_authors: List[Any]) -> List[str]:
    fams = []
    for a in found_authors:
        if isinstance(a, dict):
            fams.append(a.get("family", ""))
        elif isinstance(a, str):
            if "," in a:
                fams.append(a.split(",")[0].strip())
            else:
                toks = a.split()
                if toks:
                    fams.append(toks[-1])
    return fams


def compute_match_score(parsed: Dict[str, Any], found: Dict[str, Any]) -> float:
    title_sim = similarity(parsed.get("title", "") or "", found.get("title", "") or "")
    parsed_first = first_family_from_list(parsed.get("authors", []) or [])
    found_first = ""
    fa = found.get("authors", []) or []
    if isinstance(fa, list) and fa:
        found_first = family_list_from_found([fa[0]])[0] if fa else ""
    first_author_score = 1.0 if (parsed_first and found_first and parsed_first.lower() == found_first.lower()) else 0.0
    parsed_fams = family_list_from_parsed(parsed.get("authors", []) or [])
    found_fams = family_list_from_found(found.get("authors", []) or [])
    o
