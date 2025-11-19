# streamlit_ref_tool_final.py
"""
Full Streamlit app:
- Paste or upload PDF references
- Parse references (OpenAI optional) + fallback heuristics
- Search Crossref / PubMed / EuropePMC / SemanticScholar / OpenAlex by title
- Score matches and pick best source
- Show per-reference expandable panels comparing Found vs AI-parsed metadata
- Let user choose per-reference which metadata to export
- Export RIS / BibTeX / CSV
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
from pypdf import PdfReader

st.set_page_config(page_title="Reference → RIS (compare Found vs AI)", layout="wide")

# -------------------------
# Config / secrets
# -------------------------
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    import os
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o-mini"  # change if you need
DEFAULT_THRESHOLD = 0.65

# Simple in-memory cache (per-run)
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
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
def text_hash_for_dedupe(s: str) -> str:
    s2 = normalize_text(s.lower())
    s2 = re.sub(r"\b(19|20)\d{2}\b", "", s2)
    s2 = re.sub(r"\W+", " ", s2)
    return hashlib.sha1(s2.encode("utf-8")).hexdigest()

# -------------------------
# PDF extraction
# -------------------------
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

# -------------------------
# Clean & split
# -------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)  # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # merge single newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    """Returns list of references (keeps multiline references together)."""
    if not text:
        return []
    # Normalize and fix hyphenation
    text = text.strip()
    # If the text contains double newlines, use them as separators first
    if '\n\n' in text:
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        # Further split blocks that contain multiple numbered refs
        out = []
        for b in blocks:
            out.extend(_split_block_numbered_or_heuristic(b))
        return out
    else:
        return _split_block_numbered_or_heuristic(text)

def _split_block_numbered_or_heuristic(block: str) -> List[str]:
    # Normalize internal newlines and join broken lines
    block = clean_and_join_broken_lines(block)
    lines = [l for l in re.split(r"\n", block) if l.strip()]
    # Group lines: if a line starts with numbering -> start new ref, else continuation
    refs = []
    cur = []
    for ln in lines:
        if re.match(r"^\s*(?:\[\d+\]|\d+[\.\)])\s+", ln):
            # New numbered ref
            if cur:
                refs.append(" ".join(cur).strip())
            # remove the leading number
            ln2 = re.sub(r"^\s*(?:\[\d+\]|\d+[\.\)])\s*", "", ln)
            cur = [ln2.strip()]
        else:
            # Heuristic: if line looks like the start of a new reference (Author pattern like "Lastname Initial" and capitalized) and current line ends with '.' and contains pages/vol? We'll treat as new if very likely.
            if cur:
                cur.append(ln.strip())
            else:
                # start new anyway
                cur = [ln.strip()]
    if cur:
        refs.append(" ".join(cur).strip())
    # If still only one ref but the block contains sequences like "1. ... 2. ..." we split by numbered tokens
    if len(refs) == 1 and re.search(r"(?:\n|\s)(?:\[\d+\]|\d+[\.\)])\s+", block):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", block)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return parts
    return refs

# -------------------------
# Simple local parser (fallback)
# -------------------------
def simple_local_parse(ref_text: str) -> Dict[str,Any]:
    txt = clean_and_join_broken_lines(ref_text)
    parsed = {"authors": [], "title": "", "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""}
    # DOI detection
    m = DOI_RE.search(txt)
    if m:
        parsed["doi"] = m.group(0).rstrip(',.')
    # Year
    y = re.search(r"\b(19|20)\d{2}\b", txt)
    if y:
        parsed["year"] = y.group(0)
    # Attempt split: authors . title . journal ...
    parts = re.split(r"\.\s+", txt)
    if len(parts) >= 3:
        # assume first = authors, second = title, rest include journal/vol/pages
        parsed["authors"] = [a.strip().rstrip('.') for a in re.split(r";|, (?=[A-Z][a-z])", parts[0]) if a.strip()]
        parsed["title"] = parts[1].strip()
        # guess journal as next segment (may include vol/pages)
        rest = ". ".join(parts[2:]).strip()
        # separate pages vol by colon or numbers
        vp = re.search(r"(?P<journal>.*?)(?:\s+)(?P<vol>\d+)\s*[:(]\s*(?P<pages>[\d\-–]+)", rest)
        if vp:
            parsed["journal"] = vp.group("journal").strip().rstrip(',')
            parsed["volume"] = vp.group("vol") or ""
            parsed["pages"] = vp.group("pages") or ""
        else:
            # split at last period for journal
            parsed["journal"] = parts[2].strip()
    elif len(parts) == 2:
        parsed["title"] = parts[0].strip()
        parsed["journal"] = parts[1].strip()
    else:
        # fallback: try to extract title by quotes or italics markers (rare)
        q = re.search(r'“([^”]+)”|"([^"]+)"', txt)
        if q:
            title = q.group(1) or q.group(2)
            parsed["title"] = title.strip()
        else:
            # otherwise take first 120 chars as title fallback
            parsed["title"] = txt[:120].strip()
    # pages fallback
    pg = re.search(r"(\d{1,4}\s*[-–]\s*\d{1,4})", txt)
    if pg:
        parsed["pages"] = pg.group(1).replace(" ", "")
    return parsed

# -------------------------
# OpenAI parsing (optional)
# -------------------------
def openai_parse_reference(ref_text: str) -> Optional[Dict[str,Any]]:
    if not OPENAI_API_KEY:
        return None
    prompt = (f"You are a precise metadata extractor. Convert the following single bibliographic reference into a JSON object with these exact fields:\n"
              "- authors: array of strings (each 'Family, Given' or 'Given Family')\n"
              "- title: string\n"
              "- journal: string\n"
              "- year: string (4-digit) or empty\n"
              "- volume: string or empty\n"
              "- issue: string or empty\n"
              "- pages: string or empty\n"
              "- doi: string or empty\n\nReturn ONLY valid JSON and nothing else.\n\nReference:\n\"\"\"\n{0}\n\"\"\"".format(ref_text))
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":prompt}], "temperature": 0.0, "max_tokens": 600}
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=20)
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

# -------------------------
# Multi-source search by title
# -------------------------
def crossref_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    if not title:
        return None, None
    key = "cr:" + title[:240]
    def _fn(t):
        try:
            r = requests.get("https://api.crossref.org/works", params={"query.title": t[:240], "rows": 5}, timeout=12)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", []) or []
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
            for a in best.get("author", [])[:200]:
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
            q = f'TITLE:"{t}"'
            r = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": q, "format":"json", "pageSize":5}, timeout=12)
            r.raise_for_status()
            data = r.json().get("resultList", {}).get("result", []) or r.json().get("result", [])
            if not data:
                r2 = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={"query": t, "format":"json", "pageSize":5}, timeout=12)
                r2.raise_for_status()
                data = r2.json().get("resultList", {}).get("result", []) or r2.json().get("result", [])
                if not data:
                    return None, None
            best = None; best_score = 0.0
            for it in data:
                tfound = it.get("title","") or ""
                s = similarity(t, tfound)
                if s > best_score:
                    best_score = s; best = it
            if not best:
                return None, None
            authors = []
            auth_str = best.get("authorString","")
            if auth_str:
                for a in re.split(r";|, and |, (?=[A-Z][a-z])", auth_str):
                    a = a.strip()
                    if a:
                        toks = a.split()
                        fam = toks[-1] if toks else ""
                        giv = " ".join(toks[:-1]) if len(toks)>1 else ""
                        authors.append({"family": fam, "given": giv})
            item = {
                "title": best.get("title",""),
                "journal": best.get("journalTitle","") or "",
                "year": str(best.get("pubYear","") or ""),
                "authors": authors,
                "volume": str(best.get("journalVolume","") or ""),
                "issue": str(best.get("issue","") or ""),
                "pages": best.get("pageInfo","") or "",
                "doi": best.get("doi","") or ""
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
        s = a.strip()
        if "," in s:
            fams.append(s.split(",")[0].strip())
        else:
            toks = s.split()
            if toks:
                fams.append(toks[-1])
    return fams

def family_list_from_found(found_authors: List[Any]) -> List[str]:
    fams = []
    for a in found_authors:
        if isinstance(a, dict):
            if a.get("family"):
                fams.append(a.get("family"))
            elif a.get("name"):
                fams.append(a.get("name").split()[-1])
        else:
            toks = str(a).split()
            if toks:
                fams.append(toks[-1])
    return fams

def compute_match_score(parsed: Dict[str,Any], found: Dict[str,Any]) -> float:
    # Title sim 45%
    title_sim = similarity(parsed.get("title","") or "", found.get("title","") or "")
    # First author 20%
    parsed_first = first_family_from_list(parsed.get("authors", []) or [])
    found_first = ""
    fa = found.get("authors", []) or []
    if isinstance(fa, list) and fa:
        if isinstance(fa[0], dict):
            found_first = fa[0].get("family","") or ""
        else:
            found_first = str(fa[0]).split()[-1] if fa else ""
    first_author_score = 1.0 if (parsed_first and found_first and parsed_first.lower() == found_first.lower()) else 0.0
    # Other authors 10%
    parsed_fams = family_list_from_parsed(parsed.get("authors", []) or [])
    found_fams = family_list_from_found(found.get("authors", []) or [])
    if parsed_fams and found_fams:
        parsed_others = parsed_fams[1:] if len(parsed_fams) > 1 else []
        if parsed_others:
            matches = 0
            for pf in parsed_others:
                for ff in found_fams:
                    if pf and ff and pf.lower() == ff.lower():
                        matches += 1
                        break
            other_score = matches / max(1, len(parsed_others))
        else:
            other_score = 0.0
    else:
        other_score = 0.0
    # Year 10%
    parsed_year = str(parsed.get("year","") or "")
    found_year = str(found.get("year","") or "")
    year_score = 1.0 if (parsed_year and found_year and parsed_year == found_year) else 0.0
    # Journal 10%
    journal_score = similarity(parsed.get("journal","") or "", found.get("journal","") or "")
    # Pages 5%
    parsed_pages = str(parsed.get("pages","") or "")
    found_pages = str(found.get("pages","") or "")
    pages_score = 0.0
    if parsed_pages and found_pages:
        if parsed_pages == found_pages:
            pages_score = 1.0
        elif "-" in parsed_pages and "-" in found_pages:
            try:
                a1,a2 = parsed_pages.split("-",1)
                b1,b2 = found_pages.split("-",1)
                if int(a1) <= int(b2) and int(b1) <= int(a2):
                    pages_score = 1.0
            except:
                pages_score = 0.0
    composite = 0.45 * title_sim + 0.20 * first_author_score + 0.10 * other_score + 0.10 * year_score + 0.10 * journal_score + 0.05 * pages_score
    return composite

# -------------------------
# Converters
# -------------------------
def convert_meta_to_ris(meta: Dict[str,Any]) -> str:
    lines = []
    lines.append("TY  - JOUR")
    title = meta.get("title","") if not isinstance(meta.get("title"), list) else (meta.get("title")[0] if meta.get("title") else "")
    if title:
        lines.append(f"TI  - {title}")
    # authors
    for a in meta.get("authors", [])[:200]:
        if isinstance(a, dict):
            fam = a.get("family","")
            giv = a.get("given","")
            if fam and giv:
                lines.append(f"AU  - {fam}, {giv}")
            elif fam:
                lines.append(f"AU  - {fam}")
            else:
                lines.append(f"AU  - {a}")
        else:
            lines.append(f"AU  - {a}")
    if meta.get("journal"):
        lines.append(f"JO  - {meta.get('journal')}")
    if meta.get("volume"):
        lines.append(f"VL  - {meta.get('volume')}")
    if meta.get("issue"):
        lines.append(f"IS  - {meta.get('issue')}")
    if meta.get("pages"):
        p = str(meta.get("pages"))
        if "-" in p:
            sp, ep = p.split("-",1)
            lines.append(f"SP  - {sp}")
            lines.append(f"EP  - {ep}")
        else:
            lines.append(f"SP  - {p}")
    if meta.get("year"):
        lines.append(f"PY  - {meta.get('year')}")
    if meta.get("doi"):
        lines.append(f"DO  - {meta.get('doi')}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

def convert_meta_to_bib(meta: Dict[str,Any]) -> str:
    authors = meta.get("authors", [])
    author_str = " and ".join([(f"{a.get('family','')}, {a.get('given','')}" if isinstance(a, dict) else a) for a in authors])
    title = meta.get("title","")
    journal = meta.get("journal","")
    year = meta.get("year","")
    doi = meta.get("doi","")
    key = "ref"
    if authors and isinstance(authors[0], dict):
        key = re.sub(r"\W+","", authors[0].get("family","") + year)
    bib = f"@article{{{key},\n"
    if author_str: bib += f"  author = {{{author_str}}},\n"
    if title: bib += f"  title = {{{title}}},\n"
    if journal: bib += f"  journal = {{{journal}}},\n"
    if year: bib += f"  year = {{{year}}},\n"
    if doi: bib += f"  doi = {{{doi}}},\n"
    bib = bib.rstrip(",\n") + "\n}\n\n"
    return bib

# -------------------------
# Main processing per reference
# -------------------------
def process_reference(ref_text: str, threshold: float = DEFAULT_THRESHOLD, auto_accept: bool = True) -> Dict[str,Any]:
    parsed = None
    if OPENAI_API_KEY:
        parsed = openai_parse_reference(ref_text)
    if not parsed:
        parsed = simple_local_parse(ref_text)
    if not isinstance(parsed.get("authors", []), list):
        parsed["authors"] = [parsed.get("authors")] if parsed.get("authors") else []
    title_for_search = parsed.get("title") or ref_text[:240]

    # Query multiple sources
    candidates = []
    cr_item, _ = crossref_search_title(title_for_search)
    if cr_item:
        candidates.append(("Crossref", cr_item))
    pm_item, _ = pubmed_search_title(title_for_search)
    if pm_item:
        candidates.append(("PubMed", pm_item))
    ep_item, _ = europepmc_search_title(title_for_search)
    if ep_item:
        candidates.append(("EuropePMC", ep_item))
    ss_item, _ = semanticscholar_search_title(title_for_search)
    if ss_item:
        candidates.append(("SemanticScholar", ss_item))
    oa_item, _ = openalex_search_title(title_for_search)
    if oa_item:
        candidates.append(("OpenAlex", oa_item))

    best_item = None
    best_source = None
    best_score = 0.0
    for src, item in candidates:
        score = compute_match_score(parsed, item)
        if score > best_score:
            best_score = score
            best_item = item
            best_source = src

    choose_found = best_item is not None and best_score >= threshold and auto_accept

    # Return structure:
    return {
        "original": ref_text,
        "parsed": parsed,
        "found": best_item,
        "found_source": best_source,
        "found_score": best_score,
        "choose_found_by_default": choose_found
    }

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 Reference Finder — Compare Found Metadata vs AI-parsed (choose per-ref)")

st.markdown("""
Paste references or upload PDF(s). The app will parse each reference (OpenAI optional) and search Crossref, PubMed, Europe PMC, Semantic Scholar and OpenAlex by title.
It will show a comparison for each reference: **Found metadata** (best from searches) vs **AI-parsed metadata**. Choose which to include and export as RIS/BibTeX/CSV.
""")

col_left, col_right = st.columns([3,1])
with col_right:
    auto_accept = st.checkbox("Auto-accept found metadata when score ≥ threshold", value=True)
    threshold = st.slider("Match threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    export_format = st.selectbox("Export format", ["RIS","BibTeX","CSV"])
    st.write("---")
    if OPENAI_API_KEY:
        st.success("OpenAI key found — AI parsing enabled")
    else:
        st.info("No OpenAI key — AI parsing disabled (will use local heuristics)")

with col_left:
    mode = st.radio("Input method", ["Paste references", "Upload PDF(s)"], horizontal=True)
    raw_text = ""
    if mode == "Paste references":
        raw_text = st.text_area("Paste references here (supports numbered [1], 1., 1) etc.)", height=320)
    else:
        files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if files:
            blocks = []
            for f in files:
                with st.spinner(f"Extracting {f.name}..."):
                    txt = extract_text_from_pdf(f)
                    if txt.startswith("ERROR_PDF_EXTRACT"):
                        st.error(f"Error extracting {f.name}: {txt}")
                    else:
                        m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                        block = m.group(2) if m else txt
                        blocks.append(block)
                    time.sleep(0.05)
            raw_text = "\n\n".join(blocks)
            if raw_text:
                st.text_area("Extracted text (from PDF)", raw_text, height=200)

style, style_conf = ("Unknown","Low")
if raw_text:
    style, style_conf = ("Unknown","Low")
    try:
        style, style_conf = ("Unknown","Low")
        # quick detection
        style = "Vancouver" if re.search(r"\b(19|20)\d{2}\s*;\s*\d+", raw_text) else "Unknown"
        style_conf = "Low"
    except:
        pass
st.caption(f"Detected style hint: {style} ({style_conf})")

if st.button("Process references"):
    if not raw_text or not raw_text.strip():
        st.warning("Paste references or upload PDFs first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected after splitting.")
        st.stop()

    st.info(f"Detected {len(refs)} references — searching & parsing now. This may take some time depending on network and APIs.")
    processed = []
    progress = st.progress(0)
    for i, r in enumerate(refs, start=1):
        rec = process_reference(r, threshold=threshold, auto_accept=auto_accept)
        processed.append(rec)
        progress.progress(i / len(refs))
        time.sleep(0.12)
    st.session_state["processed_refs"] = processed
    st.success("Processing done. Review results below.")

# Review area: per-reference expandable panels
if "processed_refs" in st.session_state:
    processed = st.session_state["processed_refs"]
    st.header("Review Found vs AI-parsed metadata — choose per reference to export")
    selections = []
    seen_keys = set()
    for idx, rec in enumerate(processed, start=1):
        with st.expander(f"Reference {idx}: {rec['original'][:200]}{'...' if len(rec['original'])>200 else ''}"):
            st.markdown("**Original**")
            st.code(rec["original"])
            # Found block
            st.markdown("**Found metadata (best search result)**")
            if rec.get("found"):
                f = rec["found"]
                st.write(f"Source: {rec.get('found_source')} (score {rec.get('found_score'):.2f})")
                st.write("Title:", f.get("title",""))
                st.write("Journal:", f.get("journal",""))
                st.write("Year:", f.get("year",""))
                if f.get("doi"):
                    st.write("DOI:", f.get("doi"))
                authors_preview = []
                for a in f.get("authors", [])[:8]:
                    if isinstance(a, dict):
                        authors_preview.append(f"{a.get('family','')}, {a.get('given','')}")
                    else:
                        authors_preview.append(str(a))
                st.write("Authors (preview):", authors_preview)
                # RIS preview
                ris_found = convert_meta_to_ris({
                    "title": f.get("title",""),
                    "authors": f.get("authors", []),
                    "journal": f.get("journal",""),
                    "volume": f.get("volume",""),
                    "issue": f.get("issue",""),
                    "pages": f.get("pages",""),
                    "year": f.get("year",""),
                    "doi": f.get("doi","")
                })
                with st.expander("RIS preview — Found metadata"):
                    st.code(ris_found, language="text")
            else:
                st.warning("No search-found metadata (Crossref/PubMed/Europe PMC/Semantic Scholar/OpenAlex)")

            # AI/parsing block
            st.markdown("**AI / parsed metadata (from pasted reference)**")
            p = rec["parsed"]
            st.write("Title:", p.get("title",""))
            st.write("Journal:", p.get("journal",""))
            st.write("Year:", p.get("year",""))
            if p.get("doi"):
                st.write("DOI in pasted text:", p.get("doi"))
            st.write("Authors:", p.get("authors", []))
            ris_parsed = convert_meta_to_ris({
                "title": p.get("title",""),
                "authors": [{"family": a.split(",")[0].strip(), "given": (a.split(",")[1].strip() if "," in a else "")} if isinstance(a, str) and a else a for a in (p.get("authors") or [])],
                "journal": p.get("journal",""),
                "volume": p.get("volume",""),
                "issue": p.get("issue",""),
                "pages": p.get("pages",""),
                "year": p.get("year",""),
                "doi": p.get("doi","")
            })
            with st.expander("RIS preview — Parsed metadata"):
                st.code(ris_parsed, language="text")

            # Choose action for this ref
            default_idx = 0 if rec.get("choose_found_by_default") and rec.get("found") else 1
            choice = st.radio(f"Choose which metadata to export for reference {idx}:", ("Use found metadata (search result)","Use parsed metadata (from pasted)"), index=default_idx, key=f"choice_{idx}")
            include = st.checkbox("Include this reference in final export", value=True, key=f"include_{idx}")

            selections.append({
                "include": include,
                "choice": choice,
                "rec": rec,
                "ris_found": ris_found if rec.get("found") else "",
                "ris_parsed": ris_parsed
            })

    # After review, build export
    st.header("Export selected references")
    # Count selected
    selected_count = sum(1 for s in selections if s["include"])
    st.write(f"References selected for export: {selected_count}")

    if selected_count == 0:
        st.info("No references selected. Use the checkboxes above.")
    else:
        if st.button("Generate & Download Export"):
            ris_out = []
            bib_out = []
            csv_rows = []
            for s in selections:
                if not s["include"]:
                    continue
                rec = s["rec"]
                chosen_meta = None
                if s["choice"].startswith("Use found") and rec.get("found"):
                    # build normalized meta structure
                    f = rec["found"]
                    chosen_meta = {
                        "title": f.get("title",""),
                        "authors": f.get("authors", []),
                        "journal": f.get("journal",""),
                        "volume": f.get("volume",""),
                        "issue": f.get("issue",""),
                        "pages": f.get("pages",""),
                        "year": f.get("year",""),
                        "doi": f.get("doi","")
                    }
                else:
                    p = rec["parsed"]
                    # normalize authors to dicts
                    auths = []
                    for a in (p.get("authors") or []):
                        if isinstance(a, dict):
                            auths.append(a)
                        elif isinstance(a, str):
                            sname = a.strip()
                            if "," in sname:
                                fam, giv = [x.strip() for x in sname.split(",",1)]
                            else:
                                toks = sname.split()
                                fam = toks[-1] if toks else ""
                                giv = " ".join(toks[:-1]) if len(toks)>1 else ""
                            auths.append({"family": fam, "given": giv})
                    chosen_meta = {
                        "title": p.get("title",""),
                        "authors": auths,
                        "journal": p.get("journal",""),
                        "volume": p.get("volume",""),
                        "issue": p.get("issue",""),
                        "pages": p.get("pages",""),
                        "year": p.get("year",""),
                        "doi": p.get("doi","")
                    }
                # dedupe by DOI or text hash
                dedupe_key = (chosen_meta.get("doi") or "").lower() or text_hash_for_dedupe(rec["original"])
                if dedupe_key in text_hash_for_dedupe.__dict__.get("_seen", set()):
                    # skip duplicate
                    continue
                # mark seen (store in function attribute to persist during the button click)
                seen = text_hash_for_dedupe.__dict__.setdefault("_seen", set())
                seen.add(dedupe_key)

                ris_out.append(convert_meta_to_ris(chosen_meta))
                bib_out.append(convert_meta_to_bib(chosen_meta))
                csv_rows.append({
                    "original": rec["original"],
                    "title": chosen_meta.get("title",""),
                    "doi": chosen_meta.get("doi",""),
                    "journal": chosen_meta.get("journal",""),
                    "year": chosen_meta.get("year",""),
                    "source": rec.get("found_source") or "parsed"
                })

            if export_format == "RIS":
                final_text = "\n".join(ris_out)
                st.download_button("Download RIS", data=final_text, file_name="references.ris", mime="application/x-research-info-systems")
                with st.expander("RIS preview"):
                    st.code(final_text, language="text")
            elif export_format == "BibTeX":
                final_text = "\n".join(bib_out)
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

st.caption("Notes: OpenAI key (optional) stored in Streamlit secrets as OPENAI_API_KEY. The app queries multiple metadata sources — network/API delays apply. Adjust threshold for stricter / looser acceptance of found metadata.")
