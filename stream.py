"""
Full Streamlit app:
- Paste or upload PDF references
- Parse references with OpenAI only (no fallback)
- Search Crossref / PubMed / EuropePMC / SemanticScholar / OpenAlex by extracted title
- Score matches and pick best source
- Show per-reference expandable panels comparing Found vs AI-parsed metadata
- Let user choose per-reference which metadata to export
- Export RIS / BibTeX / CSV
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

st.set_page_config(page_title="Reference → RIS (compare Found vs AI)", layout="wide")

# -------------------------
# Config / secrets
# -------------------------
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"
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
# OpenAI parsing (required)
# -------------------------
def openai_parse_reference(ref_text: str) -> Optional[Dict[str,Any]]:
    """Parse reference using OpenAI to extract structured metadata"""
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is required. Please set OPENAI_API_KEY in Streamlit secrets.")
        return None
    
    prompt = (f"You are a precise metadata extractor. Convert the following single bibliographic reference into a JSON object with these exact fields:\n"
              "- authors: array of strings (each 'Family, Given' or 'Given Family')\n"
              "- title: string (REQUIRED - extract the main publication title)\n"
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
    except Exception as e:
        st.error(f"OpenAI parsing error: {str(e)}")
        return None

def extract_text_from_pdf(file_obj) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {str(e)}"

# -------------------------
# Multi-source search by title
# -------------------------
def crossref_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """Search Crossref by title"""
    if not title:
        return None, None
    
    key = "cr:" + title[:240]
    def _fn(t):
        try:
            params = {"query": t, "rows": 1}
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            items = data.get("message", {}).get("items", [])
            if items:
                item = items[0]
                auths = []
                for a in item.get("author", []):
                    fam = a.get("family", "")
                    giv = a.get("given", "")
                    auths.append({"family": fam, "given": giv})
                return {
                    "title": item.get("title", [""])[0] if item.get("title") else "",
                    "authors": auths,
                    "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
                    "year": str(item.get("issued", {}).get("date-parts", [[""]])[0][0]) if item.get("issued") else "",
                    "volume": str(item.get("volume", "")),
                    "issue": str(item.get("issue", "")),
                    "pages": item.get("page", ""),
                    "doi": item.get("DOI", "")
                }, None
            return None, None
        except Exception as e:
            return None, str(e)
    
    return cached(key, _fn, title)

def pubmed_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """Search PubMed by title"""
    if not title:
        return None, None
    
    key = "pm:" + title[:240]
    def _fn(t):
        try:
            params = {"db": "pubmed", "term": t, "rettype": "xml", "retmax": 1}
            r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=10)
            r.raise_for_status()
            
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r.content)
            ids = root.findall(".//Id")
            if ids:
                pmid = ids[0].text
                params2 = {"db": "pubmed", "id": pmid, "rettype": "xml"}
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
                
                return {
                    "title": title_text,
                    "authors": auths,
                    "journal": journal_text,
                    "year": year_text,
                    "volume": volume_text,
                    "issue": issue_text,
                    "pages": pages_text,
                    "doi": ""
                }, None
            return None, None
        except Exception as e:
            return None, str(e)
    
    return cached(key, _fn, title)

def europepmc_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """Search Europe PMC by title"""
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
            if results:
                item = results[0]
                auths = []
                for a in item.get("authorList", {}).get("author", []):
                    auths.append({"family": a.get("lastName", ""), "given": a.get("firstName", "")})
                return {
                    "title": item.get("title", ""),
                    "authors": auths,
                    "journal": item.get("journalTitle", ""),
                    "year": str(item.get("pubYear", "")),
                    "volume": str(item.get("journalVolume", "")),
                    "issue": str(item.get("issue", "")),
                    "pages": item.get("pageInfo", ""),
                    "doi": item.get("doi", "")
                }, None
            return None, None
        except Exception as e:
            return None, str(e)
    
    return cached(key, _fn, title)

def semanticscholar_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """Search Semantic Scholar by title"""
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
            if papers:
                item = papers[0]
                auths = [{"family": a.get("name", ""), "given": ""} for a in item.get("authors", [])]
                return {
                    "title": item.get("title", ""),
                    "authors": auths,
                    "journal": item.get("venue", ""),
                    "year": str(item.get("year", "")),
                    "volume": "",
                    "issue": "",
                    "pages": "",
                    "doi": item.get("externalIds", {}).get("DOI", "")
                }, None
            return None, None
        except Exception as e:
            return None, str(e)
    
    return cached(key, _fn, title)

def openalex_search_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """Search OpenAlex by title"""
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
            if results:
                item = results[0]
                auths = []
                for a in item.get("authorships", []):
                    author = a.get("author", {})
                    auths.append({"family": author.get("display_name", ""), "given": ""})
                return {
                    "title": item.get("title", ""),
                    "authors": auths,
                    "journal": item.get("primary_location", {}).get("source", {}).get("display_name", ""),
                    "year": str(item.get("publication_year", "")),
                    "volume": item.get("biblio", {}).get("volume", ""),
                    "issue": item.get("biblio", {}).get("issue", ""),
                    "pages": item.get("biblio", {}).get("first_page", ""),
                    "doi": item.get("doi", "").replace("https://doi.org/", "")
                }, None
            return None, None
        except Exception as e:
            return None, str(e)
    
    return cached(key, _fn, title)

# -------------------------
# Scoring & compare
# -------------------------
def first_family_from_list(authors_list: List[str]) -> str:
    """Extract first author's family name"""
    if not authors_list:
        return ""
    first = authors_list[0].strip()
    if "," in first:
        return first.split(",")[0].strip()
    toks = first.split()
    return toks[-1] if toks else ""

def family_list_from_parsed(authors_list: List[str]) -> List[str]:
    """Extract family names from parsed authors (string format)"""
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
    """Extract family names from found authors (dict format)"""
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

def compute_match_score(parsed: Dict[str,Any], found: Dict[str,Any]) -> float:
    """Compute similarity score between parsed and found metadata"""
    title_sim = similarity(parsed.get("title","") or "", found.get("title","") or "")
    
    parsed_first = first_family_from_list(parsed.get("authors", []) or [])
    found_first = ""
    fa = found.get("authors", []) or []
    if isinstance(fa, list) and fa:
        found_first = family_list_from_found([fa[0]])[0] if fa else ""
    first_author_score = 1.0 if (parsed_first and found_first and parsed_first.lower() == found_first.lower()) else 0.0
    
    parsed_fams = family_list_from_parsed(parsed.get("authors", []) or [])
    found_fams = family_list_from_found(found.get("authors", []) or [])
    if parsed_fams and found_fams:
        matches = sum(1 for pf in parsed_fams for ff in found_fams if pf.lower() == ff.lower())
        other_score = matches / max(len(parsed_fams), len(found_fams)) if max(len(parsed_fams), len(found_fams)) > 0 else 0.0
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
        pages_score = 1.0 if parsed_pages == found_pages else 0.5
    
    composite = 0.45 * title_sim + 0.20 * first_author_score + 0.10 * other_score + 0.10 * year_score + 0.10 * journal_score + 0.05 * pages_score
    return composite

# -------------------------
# Converters
# -------------------------
def convert_meta_to_ris(meta: Dict[str,Any]) -> str:
    """Convert metadata to RIS format"""
    lines = []
    lines.append("TY  - JOUR")
    
    title = meta.get("title","")
    if isinstance(title, list):
        title = title[0] if title else ""
    if title:
        lines.append(f"TI  - {title}")
    
    for a in meta.get("authors", [])[:200]:
        if isinstance(a, dict):
            family = a.get("family", "")
            given = a.get("given", "")
            au_str = f"{family}, {given}" if given else family
        else:
            au_str = str(a)
        if au_str.strip():
            lines.append(f"AU  - {au_str}")
    
    if meta.get("journal"):
        lines.append(f"JO  - {meta.get('journal')}")
    if meta.get("volume"):
        lines.append(f"VL  - {meta.get('volume')}")
    if meta.get("issue"):
        lines.append(f"IS  - {meta.get('issue')}")
    if meta.get("pages"):
        lines.append(f"SP  - {meta.get('pages')}")
    if meta.get("year"):
        lines.append(f"PY  - {meta.get('year')}")
    if meta.get("doi"):
        lines.append(f"DO  - {meta.get('doi')}")
    
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

def convert_meta_to_bib(meta: Dict[str,Any]) -> str:
    """Convert metadata to BibTeX format"""
    authors = meta.get("authors", [])
    author_list = []
    for a in authors:
        if isinstance(a, dict):
            family = a.get("family", "")
            given = a.get("given", "")
            author_list.append(f"{family}, {given}" if given else family)
        else:
            author_list.append(str(a))
    author_str = " and ".join(author_list)
    
    title = meta.get("title","")
    journal = meta.get("journal","")
    year = meta.get("year","")
    volume = meta.get("volume","")
    pages = meta.get("pages","")
    doi = meta.get("doi","")
    
    citekey = f"ref{int(time.time()*1000) % 10000}"
    bib = f"@article{{{citekey},\n"
    if author_str:
        bib += f"  author = {{{author_str}}},\n"
    if title:
        bib += f"  title = {{{title}}},\n"
    if journal:
        bib += f"  journal = {{{journal}}},\n"
    if year:
        bib += f"  year = {{{year}}},\n"
    if volume:
        bib += f"  volume = {{{volume}}},\n"
    if pages:
        bib += f"  pages = {{{pages}}},\n"
    if doi:
        bib += f"  doi = {{{doi}}},\n"
    bib = bib.rstrip(",\n") + "\n}\n\n"
    
    return bib

# -------------------------
# Main processing per reference
# -------------------------
def process_reference(ref_text: str, threshold: float = DEFAULT_THRESHOLD, auto_accept: bool = True) -> Dict[str,Any]:
    """Process a single reference: parse with OpenAI, search by extracted title"""
    
    # Parse reference with OpenAI only (no fallback)
    parsed = openai_parse_reference(ref_text)
    if not parsed:
        return {
            "original": ref_text,
            "parsed": {"authors": [], "title": "", "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""},
            "found": None,
            "found_source": None,
            "found_score": 0.0,
            "choose_found_by_default": False,
            "error": "OpenAI parsing failed"
        }
    
    if not isinstance(parsed.get("authors", []), list):
        parsed["authors"] = [parsed.get("authors")] if parsed.get("authors") else []
    
    # Use OpenAI-extracted title for searching
    title_for_search = parsed.get("title", "").strip()
    if not title_for_search:
        title_for_search = ref_text[:240]

    # Query multiple sources by title
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

    # Score and pick best match
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
st.title("📚 Reference Finder")

st.markdown("""
Paste references or upload PDF(s). OpenAI will extract the title from each reference, then search Crossref, PubMed, Europe PMC, Semantic Scholar and OpenAlex by that title.
It will show a comparison for each reference: **Found metadata** (best from searches) vs **AI-parsed metadata**. Choose which to include and export as RIS/BibTeX/CSV.
""")

col_left, col_right = st.columns([3,1])
with col_right:
    auto_accept = st.checkbox("Auto-accept found metadata when score ≥ threshold", value=True)
    threshold = st.slider("Match threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    export_format = st.selectbox("Export format", ["RIS","BibTeX","CSV"])
    st.write("---")
    if OPENAI_API_KEY:
        st.success("✓ OpenAI key found — AI parsing enabled")
    else:
        st.error("✗ OpenAI API key required — set OPENAI_API_KEY in secrets")

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
        raw_text = st.text_area("Paste one reference at a time (OpenAI will extract title)", height=320)
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
                        blocks.append(txt)
                    time.sleep(0.05)
            raw_text = "\n\n".join(blocks)
            if raw_text:
                st.text_area("Extracted text (from PDF)", raw_text, height=200)

if st.button("Process reference(s)"):
    if not raw_text or not raw_text.strip():
        st.warning("Paste a reference or upload PDFs first.")
        st.stop()

    # Split by double newlines (simple splitting by user paragraph)
    ref_blocks = [r.strip() for r in raw_text.split('\n\n') if r.strip()]
    if not ref_blocks:
        ref_blocks = [raw_text.strip()]

    st.info(f"Processing {len(ref_blocks)} reference block(s) — extracting titles with OpenAI and searching now.")
    processed = []
    progress = st.progress(0)
    
    for i, ref_block in enumerate(ref_blocks, start=1):
        rec = process_reference(ref_block, threshold=threshold, auto_accept=auto_accept)
        processed.append(rec)
        progress.progress(i / len(ref_blocks))
        time.sleep(0.12)
    
    st.session_state["processed_refs"] = processed
    st.success("Processing done. Review results below.")

# Review area: per-reference expandable panels
if "processed_refs" in st.session_state:
    processed = st.session_state["processed_refs"]
    st.header("Review Found vs AI-parsed metadata — choose per reference to export")
    selections = []
    
    for idx, rec in enumerate(processed, start=1):
        score = rec.get("found_score", 0.0)
        if score >= 0.85:
            color_badge = "🟢"
            color_label = "Excellent"
        elif score >= 0.70:
            color_badge = "🟡"
            color_label = "Good"
        elif score >= threshold:
            color_badge = "🟠"
            color_label = "Acceptable"
        else:
            color_badge = "🔴"
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
                st.warning("No search-found metadata (all sources returned no matches)")

            # AI/parsing block
            st.markdown("**AI-extracted metadata**")
            p = rec["parsed"]
            st.write("Title:", p.get("title",""))
            st.write("Journal:", p.get("journal",""))
            st.write("Year:", p.get("year",""))
            if p.get("doi"):
                st.write("DOI:", p.get("doi"))
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
            with st.expander("RIS preview — AI-extracted metadata"):
                st.code(ris_parsed, language="text")

            # Choose action for this ref
            default_idx = 0 if rec.get("choose_found_by_default") and rec.get("found") else 1
            choice = st.radio(f"Choose which metadata to export for reference {idx}:", ("Use found metadata (search result)","Use AI-extracted metadata"), index=default_idx, key=f"choice_{idx}")
            include = st.checkbox("Include this reference in final export", value=True, key=f"include_{idx}")

            selections.append({
                "include": include,
                "choice": choice,
                "rec": rec,
                "ris_found": ris_found if rec.get("found") else "",
                "ris_parsed": ris_parsed
            })

    # Export section
    st.header("Export selected references")
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
                
                if s["choice"].startswith("Use found") and rec.get("found"):
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

                ris_out.append(convert_meta_to_ris(chosen_meta))
                bib_out.append(convert_meta_to_bib(chosen_meta))
                csv_rows.append({
                    "original": rec["original"][:100],
                    "title": chosen_meta.get("title",""),
                    "doi": chosen_meta.get("doi",""),
                    "journal": chosen_meta.get("journal",""),
                    "year": chosen_meta.get("year",""),
                    "source": rec.get("found_source") or "ai-extracted"
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

st.caption("OpenAI extracts title from each reference, then searches multiple metadata sources. Set threshold to control match strictness.")
