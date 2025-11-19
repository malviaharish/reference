# streamlit_ref_tool_europemc_full.py
"""
Reference → RIS/BibTeX Streamlit app
Search & compare metadata across Crossref, PubMed (NCBI), Europe PMC (and optional SemanticScholar/OpenAlex).
If best-match score >= threshold -> use that source (and DOI). Otherwise use AI-parsed metadata.
User can review and choose which metadata to export. Exports: RIS, BibTeX, CSV.

Before running:
- Add your OpenAI key to Streamlit secrets under OPENAI_API_KEY (optional but recommended).
  .streamlit/secrets.toml:
  OPENAI_API_KEY = "sk-..."

Run:
  pip install -r requirements.txt
  streamlit run streamlit_ref_tool_europemc_full.py
"""

from typing import List, Tuple, Optional, Dict, Any
import streamlit as st
import re
import requests
import json
import io
import csv
import time
import hashlib
from difflib import SequenceMatcher

# PDF extraction
from pypdf import PdfReader

# ----------------------------
# Configuration / secrets
# ----------------------------
st.set_page_config(page_title="Reference → RIS (Crossref/PubMed/EuropePMC + AI)", layout="wide")

OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    import os
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OpenAI model (adjust if you have different access)
OPENAI_MODEL = "gpt-4o-mini"

# Match threshold (default 0.65)
DEFAULT_MATCH_THRESHOLD = 0.65

# Simple in-memory cache for API calls (per run)
API_CACHE: Dict[str, Any] = {}

def cached(key: str, fn, *args, **kwargs):
    if key in API_CACHE:
        return API_CACHE[key]
    res = fn(*args, **kwargs)
    API_CACHE[key] = res
    return res

# ----------------------------
# Utility functions
# ----------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

def hash_for_text(s: str) -> str:
    s2 = normalize_text(s.lower())
    s2 = re.sub(r"\b(19|20)\d{2}\b", "", s2)
    s2 = re.sub(r"\W+", " ", s2)
    return hashlib.sha1(s2.encode("utf-8")).hexdigest()

# ----------------------------
# PDF extraction
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {e}"

# ----------------------------
# Cleaning & splitting references
# ----------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove hyphenation across line breaks
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # Join single newlines (keep double newlines)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # Collapse extra whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    if not text:
        return []
    text = clean_and_join_broken_lines(text)

    # If numbered markers exist, split on them
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            return parts

    # Otherwise heuristic splitting
    cand = re.split(r"\.\s+(?=[A-Z\[])", text)
    results = []
    for c in cand:
        c = c.strip()
        if not c:
            continue
        if not c.endswith("."):
            c += "."
        if len(c) < 30 and results:
            results[-1] = results[-1].rstrip(".") + " " + c
        else:
            results.append(c)
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    return final

def detect_reference_style(text: str) -> Tuple[str,str]:
    if not text:
        return "Unknown","Low"
    scores = {"Vancouver":0, "APA":0, "MLA":0, "Chicago":0}
    scores["Vancouver"] += len(re.findall(r"\b(19|20)\d{2}\s*;\s*\d+", text))
    scores["APA"] += len(re.findall(r"\([1-2][0-9]{3}\)", text))
    scores["MLA"] += text.count('"') + text.count('“') + text.count('”')
    scores["Chicago"] += len(re.findall(r"^\s*\d+\.\s", text, flags=re.M))
    best = max(scores, key=scores.get)
    conf = "Low"
    if scores[best] >= 4:
        conf = "High"
    elif scores[best] >= 1:
        conf = "Medium"
    if all(v == 0 for v in scores.values()):
        return "Unknown", "Low"
    return best, conf

# ----------------------------
# Local fallback parser (simple heuristics)
# ----------------------------
def simple_local_parse(ref_text: str) -> Dict[str,Any]:
    t = clean_and_join_broken_lines(ref_text)
    parsed = {"authors": [], "title": "", "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""}
    # DOI
    m = DOI_RE.search(t)
    if m:
        parsed["doi"] = m.group(0).rstrip(',.')
    # Year
    y = re.search(r"\b(19|20)\d{2}\b", t)
    if y:
        parsed["year"] = y.group(0)
    # Try split by first period
    parts = re.split(r"\.\s+", t, maxsplit=2)
    if len(parts) >= 2:
        first, second = parts[0].strip(), parts[1].strip()
        # If first looks like authors
        if re.search(r"[A-Z][a-z]+,", first) or ("," in first and len(first.split()) <= 8):
            # Authors
            parsed["authors"] = [a.strip().rstrip('.') for a in re.split(r";|, (?=[A-Z][a-z])", first) if a.strip()]
            parsed["title"] = second.split(".")[0].strip()
            # Attempt journal
            rem = second.split(".")[1:] if "." in second else []
            parsed["journal"] = (" ".join(rem)).strip() if rem else ""
        else:
            # assume first is title
            parsed["title"] = first
            parsed["journal"] = second
    else:
        # fallback: look for colon
        if ":" in t:
            parsed["title"] = t.split(":")[0].strip()
        else:
            parsed["title"] = t[:200]
    # pages
    pg = re.search(r"(\d{1,4}\s*[-–]\s*\d{1,4})", t)
    if pg:
        parsed["pages"] = pg.group(1).replace(" ", "")
    return parsed

# ----------------------------
# OpenAI parsing
# ----------------------------
def openai_parse_reference(ref_text: str) -> Optional[Dict[str,Any]]:
    if not OPENAI_API_KEY:
        return None
    prompt = (
        "You are a precise metadata extractor. Convert the following single bibliographic reference into a JSON object with these exact fields:\n"
        "- authors: array of strings (each 'Family, Given' or 'Given Family')\n"
        "- title: string\n"
        "- journal: string\n"
        "- year: string (4-digit) or empty\n"
        "- volume: string or empty\n"
        "- issue: string or empty\n"
        "- pages: string or empty (e.g., '82-84')\n"
        "- doi: string or empty\n\nReturn ONLY valid JSON and nothing else.\n\n"
        f"Reference:\n\"\"\"\n{ref_text}\n\"\"\"\n"
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    body = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0.0, "max_tokens":600}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=20)
        r.raise_for_status()
        j = r.json()
        content = j["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {
            "authors": parsed.get("authors", []),
            "title": parsed.get("title", "").strip(),
            "journal": parsed.get("journal", "").strip(),
            "year": str(parsed.get("year","")).strip(),
            "volume": parsed.get("volume","").strip(),
            "issue": parsed.get("issue","").strip(),
            "pages": parsed.get("pages","").strip(),
            "doi": parsed.get("doi","").strip()
        }
    except Exception:
        return None

# ----------------------------
# Multi-source searches (title-only)
# ----------------------------
def crossref_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "cr:" + title[:240]
    def _fn(t):
        try:
            r = requests.get("https://api.crossref.org/works", params={"query.title": t[:240], "rows": 5}, timeout=12)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if not items:
                return None, None
            best = None; best_score = 0.0
            for it in items:
                tfound = (it.get("title") or [""])[0]
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s; best = it
            if not best:
                return None, None
            authors = []
            for a in best.get("author", [])[:50]:
                authors.append({"family": a.get("family",""), "given": a.get("given","")})
            item = {
                "title": (best.get("title") or [""])[0],
                "journal": (best.get("container-title") or [""])[0] if best.get("container-title") else "",
                "year": str(best.get("issued",{}).get("date-parts", [[None]])[0][0]) if best.get("issued") else "",
                "authors": authors,
                "volume": best.get("volume","") or "",
                "issue": best.get("issue","") or "",
                "pages": best.get("page","") or "",
                "doi": best.get("DOI","") or ""
            }
            return item, item.get("doi") or None
        except Exception:
            return None, None
    return cached(key, _fn, title)

def pubmed_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "pm:" + title[:240]
    def _fn(t):
        try:
            es = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                              params={"db":"pubmed","term":t,"retmax":5,"retmode":"json"}, timeout=12)
            es.raise_for_status()
            ids = es.json().get("esearchresult", {}).get("idlist", [])
            if not ids:
                return None, None
            best_item = None; best_score = 0.0
            for pmid in ids[:3]:
                fetch = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                                     params={"db":"pubmed","id":pmid,"retmode":"xml"}, timeout=12)
                fetch.raise_for_status()
                xml = fetch.text
                title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
                tfound = title_m.group(1).strip() if title_m else ""
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s
                    # parse metadata
                    journal_m = re.search(r"<Journal>.*?<Title>(.*?)</Title>", xml, re.S)
                    year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
                    pages_m = re.search(r"<MedlinePgn>(.*?)</MedlinePgn>", xml, re.S)
                    vol_m = re.search(r"<Volume>(.*?)</Volume>", xml)
                    issue_m = re.search(r"<Issue>(.*?)</Issue>", xml)
                    doi_m = re.search(r'<ArticleId IdType="doi">(.+?)</ArticleId>', xml)
                    authors = []
                    for m in re.finditer(r"<Author>(.*?)</Author>", xml, re.S):
                        block = m.group(1)
                        last = re.search(r"<LastName>(.*?)</LastName>", block)
                        fore = re.search(r"<ForeName>(.*?)</ForeName>", block)
                        if last:
                            authors.append({"family": last.group(1).strip(), "given": fore.group(1).strip() if fore else ""})
                    item = {
                        "title": tfound,
                        "journal": journal_m.group(1).strip() if journal_m else "",
                        "year": year_m.group(1) if year_m else "",
                        "authors": authors,
                        "volume": vol_m.group(1).strip() if vol_m else "",
                        "issue": issue_m.group(1).strip() if issue_m else "",
                        "pages": pages_m.group(1).strip() if pages_m else "",
                        "doi": doi_m.group(1).strip() if doi_m else ""
                    }
                    best_item = (item, item.get("doi") or None)
            if best_item:
                return best_item
            return None, None
        except Exception:
            return None, None
    return cached(key, _fn, title)

def europepmc_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "epmc:" + title[:240]
    def _fn(t):
        try:
            # Use title:"..." query
            # Europe PMC search REST: https://www.ebi.ac.uk/europepmc/webservices/rest/
            q = f'TITLE:"{t}"'
            r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": q, "format":"json", "pageSize":5}, timeout=12)
            r.raise_for_status()
            data = r.json().get("resultList", {}).get("result", []) or r.json().get("result", [])
            if not data:
                # fallback to general query without field
                r2 = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": t, "format":"json", "pageSize":5}, timeout=12)
                r2.raise_for_status()
                data = r2.json().get("resultList", {}).get("result", []) or r2.json().get("result", [])
                if not data:
                    return None, None
            best = None; best_score = 0.0
            for it in data:
                tfound = it.get("title","") or it.get("title", "")
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s; best = it
            if not best:
                return None, None
            # authors
            authors = []
            auth_str = best.get("authorString","")
            if auth_str:
                # split by comma or semicolon
                for a in re.split(r";|, and |, (?=[A-Z][a-z])", auth_str):
                    a = a.strip()
                    if a:
                        toks = a.split()
                        fam = toks[-1]
                        giv = " ".join(toks[:-1]) if len(toks) > 1 else ""
                        authors.append({"family": fam, "given": giv})
            item = {
                "title": best.get("title",""),
                "journal": best.get("journalTitle","") or best.get("journalTitle", ""),
                "year": str(best.get("pubYear","") or best.get("firstPublicationDate","")[:4] if best.get("firstPublicationDate") else ""),
                "authors": authors,
                "volume": str(best.get("journalVolume","") or ""),
                "issue": str(best.get("issue","") or ""),
                "pages": best.get("pageInfo","") or "",
                "doi": best.get("doi","") or best.get("doi","")
            }
            return item, item.get("doi") or None
        except Exception:
            return None, None
    return cached(key, _fn, title)

def semanticscholar_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "ss:" + title[:240]
    def _fn(t):
        try:
            r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search", params={"query": t, "limit":5, "fields":"title,year,venue,authors,externalIds"}, timeout=12)
            r.raise_for_status()
            data = r.json().get("data", []) or []
            if not data:
                return None, None
            best = None; best_score = 0.0
            for p in data:
                tfound = p.get("title","")
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s; best = p
            if not best:
                return None, None
            ext = best.get("externalIds") or {}
            doi = ext.get("DOI") or ext.get("doi") or ""
            authors = []
            for a in best.get("authors", [])[:50]:
                name = a.get("name","")
                toks = name.strip().split()
                fam = toks[-1] if toks else ""
                giv = " ".join(toks[:-1]) if len(toks)>1 else ""
                authors.append({"family": fam, "given": giv})
            item = {
                "title": best.get("title",""),
                "journal": best.get("venue","") or "",
                "year": str(best.get("year","") or ""),
                "authors": authors,
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": doi or ""
            }
            return item, item.get("doi") or None
        except Exception:
            return None, None
    return cached(key, _fn, title)

def openalex_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "oa:" + title[:240]
    def _fn(t):
        try:
            r = requests.get("https://api.openalex.org/works", params={"filter":"title.search:" + t, "per-page":5}, timeout=12)
            r.raise_for_status()
            items = r.json().get("results", []) or []
            if not items:
                return None, None
            best = None; best_score = 0.0
            for it in items:
                tfound = it.get("display_name","")
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s; best = it
            if not best:
                return None, None
            doi = best.get("ids", {}).get("doi","") or ""
            authors = []
            for a in (best.get("authorships") or [])[:50]:
                name = a.get("author", {}).get("display_name","")
                toks = name.split()
                fam = toks[-1] if toks else ""
                giv = " ".join(toks[:-1]) if len(toks)>1 else ""
                authors.append({"family": fam, "given": giv})
            item = {
                "title": best.get("display_name",""),
                "journal": (best.get("host_venue") or {}).get("display_name","") or "",
                "year": str(best.get("publication_year","") or ""),
                "authors": authors,
                "volume": str((best.get("host_venue") or {}).get("volume","") or ""),
                "issue": str((best.get("host_venue") or {}).get("issue","") or ""),
                "pages": best.get("page","") or "",
                "doi": doi or ""
            }
            return item, item.get("doi") or None
        except Exception:
            return None, None
    return cached(key, _fn, title)

# ----------------------------
# Matching / Scoring (weights per your spec)
# Title 45%, First author 20%, Other authors 10%, Year 10%, Journal 10%, Pages 5%
# ----------------------------
def first_author_from_parsed(parsed_authors: List[str]) -> str:
    if not parsed_authors:
        return ""
    first = parsed_authors[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    toks = first.split()
    return toks[-1] if toks else ""

def family_list_from_parsed(parsed_authors: List[str]) -> List[str]:
    fams = []
    for a in parsed_authors:
        s = a.strip()
        if "," in s:
            fam = s.split(",",1)[0].strip()
        else:
            toks = s.split()
            fam = toks[-1] if toks else ""
        if fam:
            fams.append(fam)
    return fams

def family_list_from_found(found_authors: List[Any]) -> List[str]:
    fams = []
    for a in found_authors:
        if isinstance(a, dict):
            fam = a.get("family","") or ""
            if fam:
                fams.append(fam)
        else:
            toks = str(a).split()
            if toks:
                fams.append(toks[-1])
    return fams

def compute_source_score(parsed: Dict[str,Any], found: Dict[str,Any]) -> float:
    # Title similarity (45%)
    title_sim = similarity(parsed.get("title","") or "", found.get("title","") or "")
    # First author (20%) exact match on family name
    parsed_first = first_author_from_parsed(parsed.get("authors", []))
    found_first = ""
    fa = found.get("authors", [])
    if isinstance(fa, list) and fa:
        if isinstance(fa[0], dict):
            found_first = fa[0].get("family","") or ""
        else:
            found_first = str(fa[0]).split()[-1] if fa else ""
    first_author_score = 1.0 if (parsed_first and found_first and parsed_first.lower() == found_first.lower()) else 0.0
    # Other authors overlap (10%) - proportion of parsed other authors found in found authors
    parsed_fams = family_list_from_parsed(parsed.get("authors", []))
    found_fams = family_list_from_found(found.get("authors", []))
    if parsed_fams and found_fams:
        # exclude first
        parsed_others = parsed_fams[1:] if len(parsed_fams) > 1 else []
        if parsed_others:
            matches = 0
            for pf in parsed_others:
                for ff in found_fams:
                    if pf and ff and pf.lower() == ff.lower():
                        matches += 1
                        break
            other_auth_score = matches / max(1, len(parsed_others))
        else:
            other_auth_score = 0.0
    else:
        other_auth_score = 0.0
    # Year (10%)
    parsed_year = str(parsed.get("year","") or "")
    found_year = str(found.get("year","") or "")
    year_score = 1.0 if (parsed_year and found_year and parsed_year == found_year) else 0.0
    # Journal similarity (10%)
    journal_score = similarity(parsed.get("journal","") or "", found.get("journal","") or "")
    # Pages (5%) exact or overlap
    parsed_pages = str(parsed.get("pages","") or "")
    found_pages = str(found.get("pages","") or "")
    pages_score = 0.0
    if parsed_pages and found_pages:
        if parsed_pages == found_pages:
            pages_score = 1.0
        elif "-" in parsed_pages and "-" in found_pages:
            a1,a2 = parsed_pages.split("-",1)
            b1,b2 = found_pages.split("-",1)
            # overlap
            try:
                if int(a1) <= int(b2) and int(b1) <= int(a2):
                    pages_score = 1.0
            except:
                pages_score = 0.0
    # Composite weighted score
    composite = 0.45 * title_sim + 0.20 * first_author_score + 0.10 * other_auth_score + 0.10 * year_score + 0.10 * journal_score + 0.05 * pages_score
    # Normalize possibility (values already 0..1) so composite is 0..1
    return composite

# ----------------------------
# Converters: to RIS, BibTeX
# ----------------------------
def convert_item_to_ris(item: Dict[str,Any]) -> str:
    lines = []
    lines.append("TY  - JOUR")
    # title
    title = item.get("title","") if not isinstance(item.get("title"), list) else (item.get("title")[0] if item.get("title") else "")
    if title:
        lines.append(f"TI  - {title}")
    # authors
    for a in item.get("authors", [])[:200]:
        if isinstance(a, dict):
            fam = a.get("family","")
            giv = a.get("given","")
            if fam and giv:
                lines.append(f"AU  - {fam}, {giv}")
            elif fam:
                lines.append(f"AU  - {fam}")
            else:
                lines.append(f"AU  - {str(a)}")
        else:
            lines.append(f"AU  - {a}")
    if item.get("journal"):
        lines.append(f"JO  - {item.get('journal')}")
    if item.get("volume"):
        lines.append(f"VL  - {item.get('volume')}")
    if item.get("issue"):
        lines.append(f"IS  - {item.get('issue')}")
    if item.get("pages"):
        p = str(item.get("pages"))
        if "-" in p:
            sp, ep = p.split("-",1)
            lines.append(f"SP  - {sp}")
            lines.append(f"EP  - {ep}")
        else:
            lines.append(f"SP  - {p}")
    if item.get("year"):
        lines.append(f"PY  - {item.get('year')}")
    if item.get("doi"):
        lines.append(f"DO  - {item.get('doi')}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

def convert_item_to_bibtex(item: Dict[str,Any]) -> str:
    authors = item.get("authors", [])
    author_str = " and ".join([(f"{a.get('family','')}, {a.get('given','')}" if isinstance(a, dict) else a) for a in authors])
    title = item.get("title","")
    journal = item.get("journal","")
    year = item.get("year","")
    doi = item.get("doi","")
    key = ""
    if authors and isinstance(authors[0], dict):
        key = re.sub(r"\W+","", authors[0].get("family","") + year)
    else:
        key = "ref" + year
    bib = f"@article{{{key},\n"
    if author_str: bib += f"  author = {{{author_str}}},\n"
    if title: bib += f"  title = {{{title}}},\n"
    if journal: bib += f"  journal = {{{journal}}},\n"
    if year: bib += f"  year = {{{year}}},\n"
    if doi: bib += f"  doi = {{{doi}}},\n"
    bib = bib.rstrip(",\n") + "\n}\n\n"
    return bib

# ----------------------------
# Process one reference: parse -> search -> score -> pick best or fallback
# ----------------------------
def process_reference(ref_text: str, threshold: float = DEFAULT_MATCH_THRESHOLD, auto_accept: bool = True) -> Dict[str,Any]:
    parsed = None
    if OPENAI_API_KEY:
        parsed = openai_parse_reference(ref_text)
    if not parsed:
        parsed = simple_local_parse(ref_text)
    # Ensure parsed authors is list of strings
    if not isinstance(parsed.get("authors", []), list):
        parsed["authors"] = [parsed.get("authors")] if parsed.get("authors") else []

    title_for_search = parsed.get("title") or re.sub(r"\s+", " ", ref_text)[:240]

    # Query all sources
    candidates = []
    # Crossref
    cr_item, cr_doi = crossref_search_title(title_for_search)
    if cr_item:
        candidates.append(("Crossref", cr_item))
    # PubMed
    pm_item, pm_doi = pubmed_search_title(title_for_search)
    if pm_item:
        candidates.append(("PubMed", pm_item))
    # Europe PMC
    ep_item, ep_doi = europepmc_search_title(title_for_search)
    if ep_item:
        candidates.append(("EuropePMC", ep_item))
    # Optional extras
    ss_item, ss_doi = semanticscholar_search_title(title_for_search)
    if ss_item:
        candidates.append(("SemanticScholar", ss_item))
    oa_item, oa_doi = openalex_search_title(title_for_search)
    if oa_item:
        candidates.append(("OpenAlex", oa_item))

    # Score them
    best_source = None
    best_item = None
    best_score = 0.0
    for src, item in candidates:
        score = compute_source_score(parsed, item)
        if score > best_score:
            best_score = score
            best_item = item
            best_source = src

    choose_found = False
    if best_item and best_score >= threshold and auto_accept:
        choose_found = True

    # Build returned record
    return {
        "original": ref_text,
        "parsed": parsed,
        "best_found": best_item,
        "best_source": best_source,
        "best_score": best_score,
        "choose_found_by_default": choose_found
    }

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("📚 Reference → RIS (Crossref / PubMed / Europe PMC + AI fallback)")

st.markdown("""
**Workflow**
1. Paste references or upload PDF(s).  
2. The app parses each reference (OpenAI when available) to extract a title.  
3. It searches **Crossref**, **PubMed**, **Europe PMC** (and optional extra sources) by title.  
4. Scores each found result (Title 45%, First author 20%, Other authors 10%, Year 10%, Journal 10%, Pages 5%).  
5. If the best score >= threshold (default 0.65) and auto-accept is on → use that metadata+DOI.  
   Otherwise the app will use AI-parsed metadata from pasted reference.  
6. You can review each reference, pick which metadata to export (search result or AI parsed), and select references to create final RIS/BibTeX/CSV.
""")

col_main, col_opts = st.columns([3,1])
with col_opts:
    auto_accept = st.checkbox("Auto-accept found metadata when score ≥ threshold", value=True)
    threshold = st.slider("Match threshold", 0.0, 1.0, DEFAULT_MATCH_THRESHOLD, 0.01)
    export_format = st.selectbox("Export format", ["RIS","BibTeX","CSV"])
    st.write("---")
    st.markdown("**OpenAI:**")
    if OPENAI_API_KEY:
        st.success("OpenAI key found — AI parsing enabled")
    else:
        st.warning("No OpenAI key found — AI parsing disabled (fallback heuristics used)")

with col_main:
    mode = st.radio("Input method", ["Paste references", "Upload PDF(s)"], horizontal=True)
    if mode == "Paste references":
        raw_text = st.text_area("Paste references here (supports numbered [1], 1., 1) etc.)", height=300)
    else:
        uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        raw_text = ""
        if uploaded_files:
            blocks = []
            for f in uploaded_files:
                with st.spinner(f"Extracting {f.name}..."):
                    txt = extract_text_from_pdf(f)
                    if txt.startswith("ERROR_PDF_EXTRACT"):
                        st.error(f"Error extracting {f.name}: {txt}")
                    else:
                        # try to find References section
                        m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                        block = m.group(2) if m else txt
                        blocks.append(block)
                    time.sleep(0.1)
            raw_text = "\n\n".join(blocks)
            if raw_text:
                st.text_area("Extracted text (from PDF)", raw_text, height=200)

style, style_conf = detect_reference_style(raw_text if 'raw_text' in locals() else "")
st.caption(f"Detected style: {style} (confidence: {style_conf})")

if st.button("Process references"):
    if not raw_text or not raw_text.strip():
        st.warning("Please paste references or upload PDF(s) first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected.")
        st.stop()

    st.info(f"Detected {len(refs)} references — processing (will query Crossref/PubMed/EuropePMC and optionally OpenAI). This may take some time.")
    processed = []
    progress = st.progress(0)
    for i, r in enumerate(refs, start=1):
        rec = process_reference(r, threshold=threshold, auto_accept=auto_accept)
        processed.append(rec)
        progress.progress(i/len(refs))
        time.sleep(0.12)
    st.session_state["processed_refs"] = processed
    st.success("Processing complete — review results below.")

# Review & choose
if "processed_refs" in st.session_state:
    processed = st.session_state["processed_refs"]
    st.header("Review results & select references to export")
    st.write("For each reference, pick whether to use the best-found metadata (if any) or the AI-parsed metadata. Tick the checkbox to include in the final export.")

    selected_ris = []
    selected_bib = []
    csv_rows = []
    seen_keys = set()

    for idx, rec in enumerate(processed, start=1):
        with st.expander(f"Reference {idx}: {rec['original'][:200]}{'...' if len(rec['original'])>200 else ''}"):
            st.markdown("**Original pasted reference**")
            st.code(rec["original"])

            st.markdown("**AI / parsed metadata (from pasted text)**")
            p = rec["parsed"]
            st.write(f"Title: {p.get('title','')}")
            st.write(f"Journal: {p.get('journal','')}")
            st.write(f"Year: {p.get('year','')}")
            st.write("Authors:", p.get("authors", []))
            if p.get("doi"):
                st.write("DOI in pasted text:", p.get("doi"))

            if rec.get("best_found"):
                st.markdown(f"**Best found metadata — {rec.get('best_source')} (score: {rec.get('best_score'):.2f})**")
                bf = rec["best_found"]
                st.write("Title:", bf.get("title",""))
                st.write("Journal:", bf.get("journal",""))
                st.write("Year:", bf.get("year",""))
                # show DOIs if any
                if bf.get("doi"):
                    st.write("DOI:", bf.get("doi"))
                # author preview
                auth_preview = []
                for a in bf.get("authors", [])[:8]:
                    if isinstance(a, dict):
                        auth_preview.append(f"{a.get('family','')}, {a.get('given','')}")
                    else:
                        auth_preview.append(str(a))
                st.write("Authors (found):", auth_preview)
            else:
                st.warning("No metadata found in Crossref/PubMed/EuropePMC/OpenAlex/SemanticScholar for this title")

            default_choice = 0 if rec.get("choose_found_by_default") and rec.get("best_found") else 1
            choice = st.radio("Choose metadata to export for this reference:", ("Use found metadata (best)", "Use parsed metadata (from pasted)"), index=default_choice, key=f"choice_{idx}")
            include = st.checkbox("Include this reference in export", value=True, key=f"include_{idx}")

            if include:
                # Determine chosen item
                if choice.startswith("Use found") and rec.get("best_found"):
                    chosen = rec["best_found"].copy()
                else:
                    # convert parsed to item shaped dict
                    parsed = rec["parsed"]
                    authors_list = []
                    for a in parsed.get("authors", [])[:200]:
                        s = a.strip()
                        if "," in s:
                            fam,giv = [x.strip() for x in s.split(",",1)]
                        else:
                            toks = s.split()
                            fam = toks[-1] if toks else ""
                            giv = " ".join(toks[:-1]) if len(toks)>1 else ""
                        if fam or giv:
                            authors_list.append({"family": fam, "given": giv})
                        else:
                            authors_list.append(s)
                    chosen = {
                        "title": parsed.get("title",""),
                        "journal": parsed.get("journal",""),
                        "year": parsed.get("year",""),
                        "authors": authors_list,
                        "volume": parsed.get("volume",""),
                        "issue": parsed.get("issue",""),
                        "pages": parsed.get("pages",""),
                        "doi": parsed.get("doi","")
                    }
                # dedupe key: DOI preferred else text hash
                dedupe_key = (chosen.get("doi") or "").lower() or hash_for_text(rec["original"])
                if dedupe_key in seen_keys:
                    st.info("This reference appears duplicated (same DOI or text). Skipping duplicate export.")
                else:
                    seen_keys.add(dedupe_key)
                    ris = convert_item_to_ris(chosen)
                    bib = convert_item_to_bibtex(chosen)
                    selected_ris.append(ris)
                    selected_bib.append(bib)
                    csv_rows.append({
                        "original": rec["original"],
                        "title": chosen.get("title",""),
                        "doi": chosen.get("doi",""),
                        "journal": chosen.get("journal",""),
                        "year": chosen.get("year",""),
                        "source": rec.get("best_source") or ("parsed")
                    })

    # Export block
    st.header("Export selected references")
    if not selected_ris:
        st.info("No references selected for export. Use the checkboxes above to include references.")
    else:
        if export_format == "RIS":
            final_text = "\n".join(selected_ris)
            st.download_button("Download RIS", data=final_text, file_name="references.ris", mime="application/x-research-info-systems")
            with st.expander("RIS preview"):
                st.code(final_text, language="text")
        elif export_format == "BibTeX":
            final_text = "\n".join(selected_bib)
            st.download_button("Download BibTeX", data=final_text, file_name="references.bib", mime="text/x-bibtex")
            with st.expander("BibTeX preview"):
                st.code(final_text, language="text")
        else:
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=["original","title","doi","journal","year","source"])
            writer.writeheader()
            for r in csv_rows:
                writer.writerow(r)
            csv_data = buf.getvalue()
            st.download_button("Download CSV", data=csv_data, file_name="references.csv", mime="text/csv")
            with st.expander("CSV preview"):
                st.code(csv_data, language="text")

st.caption("This app queries Crossref, NCBI/ PubMed, Europe PMC; optionally Semantic Scholar and OpenAlex to maximize metadata matches. OpenAI parsing is used to extract structured metadata from pasted references when required. Adjust the threshold to control acceptance of search-found metadata.")
