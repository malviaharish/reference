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


import streamlit as st
import os
import re
import json
import hashlib
import time
import requests
import io
import csv
from typing import List, Tuple, Optional, Dict, Any
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
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"  # change if you need
DEFAULT_THRESHOLD = 0.3

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
    s2 = normalize_text((s or "").lower())
    s2 = re.sub(r"\b(19|20)\d{2}\b", "", s2)
    s2 = re.sub(r"\W+", " ", s2)
    return hashlib.sha1(s2.encode("utf-8")).hexdigest()

# -------------------------
# Clean / fix spacing / split
# -------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)  # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # merge single newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def fix_spacing_and_continuous_words(ref: str) -> str:
    """Try to fix missing spaces, punctuation spacing and collapsed words in a single reference."""
    if not ref:
        return ""
    s = clean_and_join_broken_lines(ref)
    # insert space after punctuation if missing (e.g. "Smith,J" -> "Smith, J")
    s = re.sub(r'([,\.;:])([A-Za-z0-9])', r'\1 \2', s)
    # insert space between a lower/digit and an uppercase (e.g. "etAlSmith" -> "etAl Smith")
    s = re.sub(r'([a-z0-9])([A-Z][a-z])', r'\1 \2', s)
    # insert space between closing parenthesis and following word if missing
    s = re.sub(r'(\))([A-Za-z0-9])', r'\1 \2', s)
    # ensure a space after a period when followed by uppercase initial (e.g. "J.Doe" -> "J. Doe")
    s = re.sub(r'\.([A-Z])', r'. \1', s)
    # collapse multiple spaces
    s = re.sub(r'\s{2,}', ' ', s).strip()
    return s

def split_references_smart(text: str) -> List[str]:
    """Returns list of references (keeps multiline references together)."""
    if not text:
        return []
    text = text.strip()
    # If the text contains double newlines, use them as separators first
    if '\n\n' in text:
        blocks = [b.strip() for b in text.split('\n\n') if b.strip()]
        out = []
        for b in blocks:
            out.extend(_split_block_numbered_or_heuristic(b))
        return out
    else:
        return _split_block_numbered_or_heuristic(text)

def _split_block_numbered_or_heuristic(block: str) -> List[str]:
    block = clean_and_join_broken_lines(block)
    lines = [l for l in re.split(r"\n", block) if l.strip()]
    refs = []
    cur = []
    for ln in lines:
        if re.match(r"^\s*(?:\[\d+\]|\d+[\.\)])\s+", ln):
            if cur:
                refs.append(" ".join(cur).strip())
            ln2 = re.sub(r"^\s*(?:\[\d+\]|\d+[\.\)])\s*", "", ln)
            cur = [ln2.strip()]
        else:
            if cur:
                cur.append(ln.strip())
            else:
                cur = [ln.strip()]
    if cur:
        refs.append(" ".join(cur).strip())
    if len(refs) == 1 and re.search(r"(?:\n|\s)(?:\[\d+\]|\d+[\.\)])\s+", block):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", block)
        parts = [p.strip() for p in parts if p.strip()]
        if parts:
            return parts
    return refs


def openai_parse(ref_text: str) -> Dict[str, Any]:
    """Parse reference using OpenAI API"""
    client = openai.OpenAI()
    
    prompt = f"""Extract the following fields from this reference text and return as JSON:
- authors (list of author names)
- title (publication title)
- journal (journal/publication name)
- year (publication year)
- volume (volume number)
- issue (issue number)
- pages (page numbers)
- doi (DOI if present)

Reference text:
{ref_text}

Return ONLY valid JSON with these fields. Use null for missing fields."""
    
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a reference parsing assistant. Extract bibliographic information and return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    import json
    try:
        parsed = json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        parsed = {"authors": [], "title": "", "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""}
    
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
    title_sim = similarity(parsed.get("title","") or "", found.get("title","") or "")
    parsed_first = first_family_from_list(parsed.get("authors", []) or [])
    found_first = ""
    fa = found.get("authors", []) or []
    if isinstance(fa, list) and fa:
        if isinstance(fa[0], dict):
            found_first = fa[0].get("family","") or ""
        else:
            found_first = str(fa[0]).split()[-1] if fa else ""
    first_author_score = 1.0 if (parsed_first and found_first and parsed_first.lower() == found_first.lower()) else 0.0
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
    parsed_year = str(parsed.get("year","") or "")
    found_year = str(found.get("year","") or "")
    year_score = 1.0 if (parsed_year and found_year and parsed_year == found_year) else 0.0
    journal_score = similarity(parsed.get("journal","") or "", found.get("journal","") or "")
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
        key = re.sub(r"\W+","", authors[0].get("family","") + (year or ""))
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
    # Normalize & fix spacing before parsing / searching
    fixed_ref = fix_spacing_and_continuous_words(ref_text or "")
    parsed = None
    if OPENAI_API_KEY:
        parsed = openai_parse_reference(fixed_ref)
    if not parsed:
        parsed = simple_local_parse(fixed_ref)
    if not isinstance(parsed.get("authors", []), list):
        parsed["authors"] = [parsed.get("authors")] if parsed.get("authors") else []
    title_for_search = parsed.get("title") or fixed_ref[:240]

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

    return {
        "original": fixed_ref,
        "parsed": parsed,
        "found": best_item,
        "found_source": best_source,
        "found_score": best_score,
        "choose_found_by_default": choose_found
    }

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 Reference Finder")

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

    # Visible colour/score legend for the page
    st.markdown("### Match score legend")
    st.markdown(
        """
        <div style="display:flex;flex-direction:column;gap:6px;">
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="display:inline-block;width:14px;height:14px;background:#16a34a;border-radius:3px;"></span>
            <strong>Excellent</strong> — score ≥ 0.85 (🟢)
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="display:inline-block;width:14px;height:14px;background:#f59e0b;border-radius:3px;"></span>
            <strong>Good</strong> — 0.70 ≤ score &lt; 0.85 (🟡)
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="display:inline-block;width:14px;height:14px;background:#f97316;border-radius:3px;"></span>
            <strong>Acceptable</strong> — score ≥ threshold (🟠)
          </div>
          <div style="display:flex;gap:8px;align-items:center;">
            <span style="display:inline-block;width:14px;height:14px;background:#ef4444;border-radius:3px;"></span>
            <strong>Poor</strong> — score &lt; threshold (🔴)
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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
        # Color code based on match score
        score = rec.get("found_score", 0.0)
        if score >= 0.85:
            color_badge = "🟢"  # Green - excellent match
            color_label = "Excellent"
        elif score >= 0.70:
            color_badge = "🟡"  # Yellow - good match
            color_label = "Good"
        elif score >= threshold:
            color_badge = "🟠"  # Orange - acceptable match
            color_label = "Acceptable"
        else:
            color_badge = "🔴"  # Red - poor match
            color_label = "Poor"
        
        expander_title = f"{color_badge} Reference {idx}: {rec['original'][:160]}{'...' if len(rec['original'])>160 else ''} [{color_label} {score:.2f}]"
        
        with st.expander(expander_title):
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

# test_pubmed_search_title.py
import pytest
from unittest.mock import patch, MagicMock
import streamlit_ref_tool_final


@pytest.fixture
def sample_xml_response():
    """Mock XML response from PubMed efetch"""
    return """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <Article>
      <ArticleTitle>Test Article Title Here</ArticleTitle>
      <Journal>
        <Title>Test Journal</Title>
      </Journal>
      <PubDate>
        <Year>2020</Year>
      </PubDate>
      <Pagination>
        <MedlinePgn>123-145</MedlinePgn>
      </Pagination>
      <Volume>42</Volume>
      <Issue>3</Issue>
      <AuthorList>
        <Author>
          <LastName>Smith</LastName>
          <ForeName>John</ForeName>
        </Author>
        <Author>
          <LastName>Doe</LastName>
          <ForeName>Jane</ForeName>
        </Author>
      </AuthorList>
      <ArticleIdList>
        <ArticleId IdType="doi">10.1234/test.2020.123</ArticleId>
      </ArticleIdList>
    </Article>
  </PubmedArticle>
</PubmedArticleSet>"""


def test_pubmed_search_title_with_valid_title(sample_xml_response):
    """Test successful search with valid title"""
    with patch('streamlit_ref_tool_final.requests.get') as mock_get:
        # Mock esearch response
        esearch_response = MagicMock()
        esearch_response.json.return_value = {
            "esearchresult": {"idlist": ["12345678"]}
        }
        # Mock efetch response
        efetch_response = MagicMock()
        efetch_response.text = sample_xml_response
        
        mock_get.side_effect = [esearch_response, efetch_response]
        
        result, doi = streamlit_ref_tool_final.pubmed_search_title("Test Article Title")
        
        assert result is not None
        assert result["title"] == "Test Article Title Here"
        assert result["journal"] == "Test Journal"
        assert result["year"] == "2020"
        assert result["volume"] == "42"
        assert result["issue"] == "3"
        assert result["pages"] == "123-145"
        assert result["doi"] == "10.1234/test.2020.123"
        assert len(result["authors"]) == 2
        assert result["authors"][0]["family"] == "Smith"
        assert result["authors"][0]["given"] == "John"


def test_pubmed_search_title_empty_string():
    """Test with empty title"""
    result, doi = streamlit_ref_tool_final.pubmed_search_title("")
    assert result is None
    assert doi is None


def test_pubmed_search_title_none_title():
    """Test with None title"""
    result, doi = streamlit_ref_tool_final.pubmed_search_title(None)
    assert result is None
    assert doi is None


def test_pubmed_search_title_no_results():
    """Test when esearch returns no results"""
    with patch('streamlit_ref_tool_final.requests.get') as mock_get:
        esearch_response = MagicMock()
        esearch_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = esearch_response
        
        result, doi = streamlit_ref_tool_final.pubmed_search_title("Nonexistent Article")
        
        assert result is None
        assert doi is None


def test_pubmed_search_title_api_exception():
    """Test handling of API exceptions"""
    with patch('streamlit_ref_tool_final.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")
        
        result, doi = streamlit_ref_tool_final.pubmed_search_title("Some Title")
        
        assert result is None
        assert doi is None


def test_pubmed_search_title_multiple_results_picks_best_similarity(sample_xml_response):
    """Test that function picks best match by similarity when multiple results"""
    xml_poor = """<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <Article>
      <ArticleTitle>Completely Different Article</ArticleTitle>
      <Journal><Title>J</Title></Journal>
      <PubDate><Year>2019</Year></PubDate>
      <Pagination><MedlinePgn>1-10</MedlinePgn></Pagination>
      <Volume>1</Volume>
      <Issue>1</Issue>
    </Article>
  </PubmedArticle>
</PubmedArticleSet>"""
    
    with patch('streamlit_ref_tool_final.requests.get') as mock_get:
        esearch_response = MagicMock()
        esearch_response.json.return_value = {
            "esearchresult": {"idlist": ["111", "222"]}
        }
        efetch_poor = MagicMock()
        efetch_poor.text = xml_poor
        efetch_good = MagicMock()
        efetch_good.text = sample_xml_response
        
        mock_get.side_effect = [esearch_response, efetch_poor, efetch_good]
        
        result, doi = streamlit_ref_tool_final.pubmed_search_title("Test Article Title Here")
        
        # Should pick the second result (better similarity)
        assert result is not None
        assert result["title"] == "Test Article Title Here"


def test_pubmed_search_title_caching():
    """Test that results are cached"""
    with patch('streamlit_ref_tool_final.requests.get') as mock_get:
        esearch_response = MagicMock()
        esearch_response.json.return_value = {"esearchresult": {"idlist": []}}
        mock_get.return_value = esearch_response
        
        # Clear cache first
        streamlit_ref_tool_final.API_CACHE.clear()
        
        # First call
        streamlit_ref_tool_final.pubmed_search_title("Cached Title")
        call_count_1 = mock_get.call_count
        
        # Second call (should use cache)
        streamlit_ref_tool_final.pubmed_search_title("Cached Title")
        call_count_2 = mock_get.call_count
        
        # Should not make additional API calls
        assert call_count_2 == call_count_1



