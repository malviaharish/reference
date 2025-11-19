# streamlit_ref_tool_openai.py
"""
Reference → DOI → RIS/BibTeX Streamlit app (OpenAI-assisted parsing)
Search order: Crossref (title) -> PubMed (title/full) -> Semantic Scholar
If search metadata matches pasted reference (authors & year & journal similarity) => use DOI/metadata
Else => parse pasted reference with OpenAI -> convert parsed metadata to RIS
"""

import os
import re
import time
import json
import csv
import io
import hashlib
import difflib
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader

# -----------------------
# Config / secrets
# -----------------------
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    # also check env fallback
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# -------------
# Utilities
# -------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def sim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# -------------
# PDF extraction
# -------------
def extract_text_from_pdf(uploaded) -> str:
    try:
        reader = PdfReader(uploaded)
        pages = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                pages.append(t)
        return "\n".join(pages)
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {e}"

# -------------
# Clean & split references
# -------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)  # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)  # join single newlines
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    if not text:
        return []
    text = clean_and_join_broken_lines(text)
    # numeric markers
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            return parts
    # heuristic split
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

# -----------------------
# Reference style detection
# -----------------------
def detect_reference_style(text: str) -> Tuple[str, str]:
    text_lower = text.lower() if text else ""
    scores = {"Vancouver":0, "APA":0, "MLA":0, "Chicago":0}
    # Vancouver cues: semicolon with year "2005;17" or year;vol:pages
    scores["Vancouver"] += len(re.findall(r"\b(19|20)\d{2}\s*;\s*\d+", text))
    # APA cues: (Year) in parentheses after authors
    scores["APA"] += len(re.findall(r"\([1-2][0-9]{3}\)", text))
    # MLA cues: quotes around titles
    scores["MLA"] += text.count('"') + text.count('“') + text.count('”')
    # Chicago: footnote numeric patterns maybe
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

# -----------------------
# OpenAI-based parser (structured JSON extraction)
# -----------------------
def openai_parse_reference(ref_text: str) -> Optional[Dict[str, Any]]:
    """
    Use the OpenAI Chat Completions (or standard completions) endpoint to
    convert a pasted reference into JSON with keys:
    authors (list of strings), title, journal, year, volume, issue, pages, doi
    """
    if not OPENAI_API_KEY:
        return None

    prompt = f"""You are a precise metadata extractor. Convert the following single bibliographic reference into a JSON object with these exact fields:
- authors: array of strings (each "Family, Given" or "Family GivenInitials")
- title: string
- journal: string
- year: string (4-digit) or empty
- volume: string or empty
- issue: string or empty
- pages: string or empty (e.g., "82-84")
- doi: string or empty

Return ONLY valid JSON. Do not include any explanation.

Reference:
\"\"\"{ref_text}\"\"\"

If a field cannot be determined, set it to an empty string or empty list for authors.
"""

    # Use Chat Completions API (OpenAI). We'll call v1/chat/completions with gpt-4o-mini or gpt-4o.
    # Fallback to text-davinci-003 style if needed.
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "gpt-4o-mini",  # if not available, user can change to a preferred model in secrets
        "messages": [{"role":"user","content":prompt}],
        "temperature": 0.0,
        "max_tokens": 600
    }
    try:
        resp = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, data=json.dumps(data), timeout=20)
        resp.raise_for_status()
        j = resp.json()
        # Extract assistant content
        content = j["choices"][0]["message"]["content"]
        # Parse JSON (assistant should respond with JSON)
        parsed_json = json.loads(content)
        # Normalize fields and return
        out = {
            "authors": parsed_json.get("authors", []),
            "title": parsed_json.get("title", "").strip(),
            "journal": parsed_json.get("journal", "").strip(),
            "year": str(parsed_json.get("year","")).strip(),
            "volume": parsed_json.get("volume","").strip(),
            "issue": parsed_json.get("issue","").strip(),
            "pages": parsed_json.get("pages","").strip(),
            "doi": parsed_json.get("doi","").strip()
        }
        return out
    except Exception as e:
        # If OpenAI call/parsing fails, return None
        return None

# -----------------------
# Light fallback parser (if OpenAI not available)
# -----------------------
def fallback_local_parse(ref_text: str) -> Dict[str, Any]:
    # Simple heuristics to extract fields (less accurate)
    parsed = {"authors": [], "title": "", "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""}
    text = clean_and_join_broken_lines(ref_text)
    # doi
    m = DOI_RE.search(text)
    if m:
        parsed["doi"] = m.group(0).rstrip(',.')
    # year
    ym = re.search(r"\b(19|20)\d{2}\b", text)
    if ym:
        parsed["year"] = ym.group(0)
    # split at first period
    parts = re.split(r"\.\s+", text, maxsplit=1)
    if len(parts) >= 2:
        first = parts[0].strip()
        rest = parts[1].strip()
        # authors if contains comma and initials
        if re.search(r"\b[A-Z][a-z]+,?\s+[A-Z]\.?", first) or ("," in first and len(first.split()) <= 6):
            parsed["authors"] = [a.strip().rstrip('.') for a in re.split(r";|, (?=[A-Z][a-z])", first) if a.strip()]
            title_guess, journal_guess = split_title_and_journal(rest)
            parsed["title"] = title_guess
            parsed["journal"] = journal_guess
        else:
            # maybe "Title. Journal. Year;Vol:Pages"
            title_guess, journal_guess = split_title_and_journal(text)
            parsed["title"] = title_guess
            parsed["journal"] = journal_guess
    else:
        title_guess, journal_guess = split_title_and_journal(text)
        parsed["title"] = title_guess
        parsed["journal"] = journal_guess
    # pages / vol
    vp = re.search(r"(?P<vol>\d+)\s*\(?\s*(?P<iss>\d+)?\s*\)?\s*:\s*(?P<pages>[\d\-–]+)", text)
    if vp:
        parsed["volume"] = vp.group("vol") or ""
        parsed["issue"] = vp.group("iss") or ""
        parsed["pages"] = vp.group("pages") or ""
    else:
        # alternate :pages only
        pm = re.search(r":\s*(\d+[-–]\d+|\d+)\b", text)
        if pm:
            parsed["pages"] = pm.group(1)
    return parsed

# Helper used by fallback_local_parse & openai fallback
def split_title_and_journal(ref_text: str) -> Tuple[str,str]:
    text = " ".join(ref_text.split())
    # use year boundary if present
    ym = re.search(r"\b(19|20)\d{2}\b", text)
    left = text[:ym.start()] if ym else text
    parts = [p.strip() for p in left.split(".") if p.strip()]
    if len(parts) <= 1:
        # nothing to split
        if len(parts) == 1:
            return parts[0], ""
        return "", ""
    title = parts[0]
    journal = " ".join(parts[1:])
    journal = re.sub(r"\b\d+.*$", "", journal).strip()
    return title.rstrip("."), journal.rstrip(".")

# -----------------------
# Crossref / PubMed / Semantic Scholar searches (title-based)
# -----------------------
def crossref_search_by_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        if not title:
            return None, None
        params = {"query.title": title[:240], "rows": 3}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=12)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None, None
        # pick best by title similarity
        best = None
        best_score = 0.0
        for it in items:
            t = (it.get("title") or [""])[0]
            s = sim(title, t)
            if s > best_score:
                best_score = s
                best = it
        if best:
            return best, best.get("DOI")
    except Exception:
        return None, None
    return None, None

def pubmed_search_by_title(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        if not title:
            return None, None
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db":"pubmed", "term": title, "retmode":"json", "retmax":3}
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None, None
        # fetch top 3, choose best by title similarity
        best_item = None
        best_score = 0.0
        for pmid in ids[:3]:
            item, doi = pubmed_fetch(pmid)
            if item:
                t = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else item.get("title","")
                s = sim(title, t)
                if s > best_score:
                    best_score = s
                    best_item = (item, doi)
        if best_item:
            return best_item
    except Exception:
        return None, None
    return None, None

def pubmed_fetch(pmid: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r = requests.get(fetch, params={"db":"pubmed", "id": pmid, "retmode":"xml"}, timeout=12)
        r.raise_for_status()
        xml = r.text
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
        title = title_m.group(1).strip() if title_m else ""
        journal_m = re.search(r"<Journal>.*?<Title>(.*?)</Title>", xml, re.S)
        journal = journal_m.group(1).strip() if journal_m else ""
        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
        year = int(year_m.group(1)) if year_m else 0
        authors = []
        for m in re.finditer(r"<Author>(.*?)</Author>", xml, re.S):
            block = m.group(1)
            last = re.search(r"<LastName>(.*?)</LastName>", block)
            fore = re.search(r"<ForeName>(.*?)</ForeName>", block)
            if last:
                authors.append({"family": last.group(1).strip(), "given": fore.group(1).strip() if fore else ""})
        pages_m = re.search(r"<MedlinePgn>(.*?)</MedlinePgn>", xml)
        pages = pages_m.group(1).strip() if pages_m else ""
        doi_m = re.search(r'<ArticleId IdType="doi">(.+?)</ArticleId>', xml)
        doi = doi_m.group(1).strip() if doi_m else ""
        vol_m = re.search(r"<Volume>(.*?)</Volume>", xml)
        vol = vol_m.group(1).strip() if vol_m else ""
        issue_m = re.search(r"<Issue>(.*?)</Issue>", xml)
        issue = issue_m.group(1).strip() if issue_m else ""
        item = {
            "title":[title],
            "container-title":[journal],
            "author": authors,
            "issued": {"date-parts": [[year]]},
            "volume": vol,
            "issue": issue,
            "page": pages,
            "DOI": doi
        }
        return item, doi or None
    except Exception:
        return None, None

def semantic_scholar_search(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        if not title:
            return None, None
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": title[:300], "limit": 3, "fields": "title,authors,year,externalIds,venue"}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("data", [])
        if not data:
            return None, None
        best = None; best_score = 0.0
        for p in data:
            t = p.get("title","")
            s = sim(title, t)
            if s > best_score:
                best_score = s; best = p
        if best:
            ext = best.get("externalIds") or {}
            doi = ext.get("DOI") or ext.get("DOI:")
            item = {
                "title":[best.get("title","")],
                "author":[{"family": a.get("name","").split()[-1], "given": " ".join(a.get("name","").split()[:-1])} for a in best.get("authors",[])],
                "issued":{"date-parts":[[best.get("year") or 0]]},
                "container-title":[best.get("venue","")],
                "DOI": doi or ""
            }
            return item, doi or None
    except Exception:
        return None, None
    return None, None

# -----------------------
# Converters
# -----------------------
def convert_item_to_ris(item: Dict[str, Any]) -> str:
    if not item:
        return ""
    lines = []
    typemap = {"journal-article":"JOUR", "book":"BOOK", "book-chapter":"CHAP"}
    ty = typemap.get(item.get("type",""), "GEN")
    lines.append(f"TY  - {ty}")
    # title may be list or string
    title = item.get("title",[item.get("title","")])[0] if isinstance(item.get("title"), list) else item.get("title","")
    if title:
        lines.append(f"TI  - {title}")
    for a in item.get("author", [])[:50]:
        if isinstance(a, dict):
            lines.append(f"AU  - {a.get('family','')}, {a.get('given','')}")
        else:
            lines.append(f"AU  - {a}")
    if item.get("container-title"):
        ct = item['container-title'][0] if isinstance(item['container-title'], list) else item['container-title']
        if ct:
            lines.append(f"JO  - {ct}")
    if item.get("volume"):
        lines.append(f"VL  - {item.get('volume')}")
    if item.get("issue"):
        lines.append(f"IS  - {item.get('issue')}")
    if item.get("page"):
        p = item["page"]
        if "-" in p:
            sp, ep = p.split("-",1)
            lines.append(f"SP  - {sp}")
            lines.append(f"EP  - {ep}")
        else:
            lines.append(f"SP  - {p}")
    if item.get("issued", {}).get("date-parts"):
        lines.append(f"PY  - {item['issued']['date-parts'][0][0]}")
    if item.get("DOI"):
        lines.append(f"DO  - {item.get('DOI')}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

def convert_item_to_bibtex(item: Dict[str, Any]) -> str:
    if not item:
        return ""
    first = (item.get("author") or [{}])[0]
    fam = first.get("family","") if isinstance(first, dict) else str(first)
    year = ""
    if item.get("issued",{}).get("date-parts"):
        year = str(item["issued"]["date-parts"][0][0])
    key = re.sub(r"\W+","", fam + year) or "ref"
    authors = " and ".join([(f"{a.get('family','')}, {a.get('given','')}" if isinstance(a, dict) else str(a)) for a in (item.get("author") or [])])
    title = item.get("title",[item.get("title","")])[0] if item.get("title") else ""
    journal = item.get("container-title",[item.get("container-title","")])[0] if item.get("container-title") else ""
    doi = item.get("DOI","")
    bib = f"@article{{{key},\n"
    if authors: bib += f"  author = {{{authors}}},\n"
    if title: bib += f"  title = {{{title}}},\n"
    if journal: bib += f"  journal = {{{journal}}},\n"
    if year: bib += f"  year = {{{year}}},\n"
    if doi: bib += f"  doi = {{{doi}}},\n"
    bib = bib.rstrip(",\n") + "\n}\n\n"
    return bib

def parsed_to_item_like(parsed: Dict[str, Any]) -> Dict[str, Any]:
    # Convert parsed dict (authors list, title,... ) into Crossref-like item
    item = {
        "title": [parsed.get("title","")],
        "author": [],
        "container-title":[parsed.get("journal","")],
        "issued": {"date-parts":[[int(parsed["year"])]]} if parsed.get("year") and str(parsed.get("year")).isdigit() else {"date-parts":[[0]]},
        "volume": parsed.get("volume",""),
        "issue": parsed.get("issue",""),
        "page": parsed.get("pages",""),
        "DOI": parsed.get("doi","")
    }
    for a in parsed.get("authors", [])[:50]:
        # heuristics: "Family, Given" or "Family GivenInitials" or "Given Family"
        if "," in a:
            fam, giv = [p.strip() for p in a.split(",",1)]
        else:
            toks = a.split()
            if len(toks) == 1:
                fam = toks[0]; giv = ""
            else:
                fam = toks[-1]; giv = " ".join(toks[:-1])
        item["author"].append({"family": fam, "given": giv})
    return item

# -----------------------
# Deduplication key
# -----------------------
def canonicalize_for_dedupe(item: Optional[Dict[str,Any]], ref_text: str) -> str:
    if item and item.get("DOI"):
        return item["DOI"].lower()
    s = re.sub(r"\W+", " ", ref_text.lower())
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Reference → RIS (OpenAI parser)", layout="wide")
st.title("📚 Reference → DOI → RIS (OpenAI-assisted parsing + Crossref/PubMed/SemanticScholar)")

st.markdown("""
**Workflow**
1. Detect reference style.
2. Search by **title** (Crossref → PubMed → Semantic Scholar).
3. If a search result is found, **cross-check** result vs pasted reference (authors & year & journal).
   - If match (above threshold) → use DOI & metadata.
   - If mismatch → use OpenAI-parsed metadata from the pasted reference and convert that to RIS.
4. Let user review/edit per reference. Export RIS/BibTeX/CSV.
""")

# Inputs
mode = st.radio("Input method", ["Paste references", "Upload PDF(s)"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references here (supports numbered forms like [1], 1., 1) etc.)", height=350)
else:
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        parts = []
        for f in uploaded:
            with st.spinner(f"Extracting {f.name}..."):
                txt = extract_text_from_pdf(f)
                if txt.startswith("ERROR_PDF_EXTRACT"):
                    st.error(f"Error extracting {f.name}: {txt}")
                else:
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                    block = m.group(2) if m else txt
                    parts.append(block)
        raw_text = "\n\n".join(parts)
        if raw_text:
            st.text_area("Extracted text (from PDF)", raw_text, height=200)

style, style_conf = detect_reference_style(raw_text)
st.metric("Detected reference style", style)
st.metric("Style confidence", style_conf)

auto_accept = st.checkbox("Auto-accept matched search metadata when similarity >= threshold", value=True)
threshold = st.slider("Similarity threshold for auto-accept (0-1)", min_value=0.0, max_value=1.0, value=0.70, step=0.01)
export_format = st.selectbox("Export format", ["RIS", "BibTeX", "CSV"], index=0)

if OPENAI_API_KEY:
    st.info("OpenAI API key found in secrets — OpenAI parsing enabled.")
else:
    st.warning("No OpenAI API key found in streamlit secrets or environment. OpenAI parsing disabled; fallback local parser will be used if needed.")

if st.button("Process references"):
    if not raw_text.strip():
        st.warning("Please paste references or upload PDFs first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected.")
        st.stop()
    st.success(f"Detected {len(refs)} references.")

    results = []
    seen_keys = set()
    progress = st.progress(0)
    status = st.empty()

    # Main processing loop
    for i, ref in enumerate(refs, start=1):
        status.text(f"Processing {i}/{len(refs)}")
        ref_clean = clean_and_join_broken_lines(ref)
        parsed_by_openai = None
        if OPENAI_API_KEY:
            parsed_by_openai = openai_parse_reference(ref_clean)
        if not parsed_by_openai:
            parsed_by_openai = fallback_local_parse(ref_clean)

        # Title-based search (Crossref -> PubMed -> Semantic Scholar)
        title_for_search = parsed_by_openai.get("title") or ref_clean[:240]
        item = None; doi = None; source = None

        # Crossref
        cr_item, cr_doi = crossref_search_by_title(title_for_search)
        if cr_item and cr_doi:
            item, doi, source = cr_item, cr_doi, "Crossref"
        else:
            # PubMed
            pm_item, pm_doi = pubmed_search_by_title(title_for_search)
            if pm_item and pm_doi:
                item, doi, source = pm_item, pm_doi, "PubMed"
            else:
                # Semantic Scholar fallback
                ss_item, ss_doi = semantic_scholar_search(title_for_search)
                if ss_item:
                    item, doi, source = ss_item, ss_doi, "SemanticScholar"

        # If item found, compute cross-check between item metadata and pasted reference (authors & year & journal)
        use_search_metadata = False
        match_score = 0.0
        if item:
            # extract found values
            found_title = (item.get("title") or [item.get("title","")])[0] if isinstance(item.get("title"), list) else item.get("title","")
            found_journal = ""
            if item.get("container-title"):
                found_journal = item.get("container-title")[0] if isinstance(item.get("container-title"), list) else item.get("container-title","")
            found_year = ""
            if item.get("issued", {}).get("date-parts"):
                found_year = str(item["issued"]["date-parts"][0][0])
            # authors list (family names)
            found_authors = []
            for a in item.get("author", [])[:10]:
                if isinstance(a, dict):
                    if a.get("family"):
                        found_authors.append(a.get("family"))
                    elif a.get("name"):
                        found_authors.append(a.get("name").split()[-1])
                else:
                    found_authors.append(str(a).split()[-1])
            # compute matching metrics with parsed_by_openai (pasted)
            pasted_authors = parsed_by_openai.get("authors", []) or []
            # compare author family names overlap
            def fams_from_strings(auth_list):
                fams = []
                for s in auth_list:
                    if "," in s:
                        fam = s.split(",")[0].strip()
                    else:
                        toks = s.split()
                        fam = toks[-1] if toks else ""
                    if fam:
                        fams.append(fam)
                return fams
            pasted_fams = fams_from_strings(pasted_authors)
            # compute author overlap ratio
            overlap = 0
            if pasted_fams and found_authors:
                for pf in pasted_fams:
                    for ff in found_authors:
                        if pf.lower() == ff.lower():
                            overlap += 1
                author_score = overlap / max(1, len(pasted_fams))
            else:
                author_score = 0.0
            # year similarity
            pasted_year = parsed_by_openai.get("year","") or ""
            year_score = 1.0 if (pasted_year and found_year and pasted_year == found_year) else 0.0
            # journal similarity
            journal_score = sim(parsed_by_openai.get("journal","") or "", found_journal or "")
            # composite match score (weights)
            match_score = 0.5 * author_score + 0.3 * year_score + 0.2 * journal_score
            use_search_metadata = match_score >= threshold and auto_accept

        # prepare result record
        key = canonicalize_for_dedupe(item, ref_clean)
        duplicate = key in seen_keys
        if not duplicate:
            seen_keys.add(key)
        results.append({
            "original": ref,
            "parsed_openai": parsed_by_openai,
            "found_item": item,
            "found_doi": doi,
            "found_source": source,
            "match_score": match_score,
            "use_search_metadata_default": use_search_metadata,
            "duplicate": duplicate
        })
        progress.progress(i/len(refs))
        time.sleep(0.12)

    status.empty()
    progress.empty()

    # Interactive review
    st.header("Review & decide per reference")
    choices = []
    for idx, r in enumerate(results, start=1):
        with st.expander(f"Ref {idx}: {r['original'][:200]}{'...' if len(r['original'])>200 else ''}"):
            if r["duplicate"]:
                st.info("Duplicate detected.")
            # show found metadata if any
            if r["found_item"]:
                st.markdown(f"**Search result found ({r['found_source']}) — default match score: {r['match_score']:.2f}**")
                ft = (r["found_item"].get("title") or [""])[0] if isinstance(r["found_item"].get("title"), list) else r["found_item"].get("title","")
                st.write("Found title:", ft)
                if r["found_doi"]:
                    st.write("Found DOI:", r["found_doi"])
                if r["found_item"].get("container-title"):
                    st.write("Found journal:", r["found_item"].get("container-title")[0] if isinstance(r["found_item"].get("container-title"), list) else r["found_item"].get("container-title",""))
                # authors preview
                if r["found_item"].get("author"):
                    preview = []
                    for a in (r["found_item"].get("author") or [])[:6]:
                        if isinstance(a, dict):
                            preview.append(f"{a.get('family','')} {a.get('given','')}")
                        else:
                            preview.append(str(a))
                    st.write("Found authors (preview):", ", ".join(preview))
            else:
                st.warning("No search metadata found.")

            # show OpenAI parsed metadata
            st.markdown("**OpenAI parsed (from pasted text)**")
            p = r["parsed_openai"]
            st.write("Authors:", p.get("authors"))
            st.write("Title:", p.get("title"))
            st.write("Journal:", p.get("journal"))
            st.write("Year:", p.get("year"))
            st.write("Volume/Issue/Pages:", f"{p.get('volume')}/{p.get('issue')}/{p.get('pages')}")
            if p.get("doi"):
                st.write("DOI (in pasted text):", p.get("doi"))

            # default selection
            default_index = 0 if r["use_search_metadata_default"] else 1
            action = st.radio(f"Choose action for ref {idx}", ("Use search metadata (and DOI)","Use pasted reference → Use OpenAI parsed metadata"), index=default_index, key=f"action_{idx}")
            include = st.checkbox(f"Include this reference in export", value=True, key=f"include_{idx}")

            choices.append({
                "include": include,
                "use_search": action == "Use search metadata (and DOI)",
                "record": r
            })

    # Build outputs
    ris_blocks = []
    bib_blocks = []
    csv_rows = []
    for c in choices:
        if not c["include"]:
            continue
        r = c["record"]
        if c["use_search"] and r["found_item"]:
            # use found_item
            ris_blocks.append(convert_item_to_ris(r["found_item"]))
            bib_blocks.append(convert_item_to_bibtex(r["found_item"]))
            csv_rows.append({
                "original": r["original"],
                "title": (r["found_item"].get("title") or [""])[0] if isinstance(r["found_item"].get("title"), list) else r["found_item"].get("title",""),
                "doi": r.get("found_doi") or "",
                "journal": r["found_item"].get("container-title",[r["parsed_openai"].get("journal","")])[0] if r["found_item"].get("container-title") else r["parsed_openai"].get("journal",""),
                "year": r["found_item"].get("issued",{}).get("date-parts",[[r["parsed_openai"].get("year","")]])[0][0] if r["found_item"].get("issued") else r["parsed_openai"].get("year",""),
                "source": r.get("found_source") or "search"
            })
        else:
            # use parsed_openai
            parsed = r["parsed_openai"]
            item_like = parsed_to_item_like(parsed)
            ris_blocks.append(convert_item_to_ris(item_like))
            bib_blocks.append(convert_item_to_bibtex(item_like))
            csv_rows.append({
                "original": r["original"],
                "title": parsed.get("title",""),
                "doi": parsed.get("doi",""),
                "journal": parsed.get("journal",""),
                "year": parsed.get("year",""),
                "source": "parsed"
            })

    # Export UI
    st.header("Export Results")
    cnt = len(ris_blocks)
    st.success(f"Prepared {cnt} entries for export.")

    if export_format == "RIS":
        final = "".join(ris_blocks)
        if final.strip():
            st.download_button("Download RIS", data=final, file_name="references.ris", mime="application/x-research-info-systems")
            with st.expander("RIS preview"):
                st.code(final, language="text")
        else:
            st.error("No RIS generated.")
    elif export_format == "BibTeX":
        final = "".join(bib_blocks)
        if final.strip():
            st.download_button("Download BibTeX", data=final, file_name="references.bib", mime="text/x-bibtex")
            with st.expander("BibTeX preview"):
                st.code(final, language="text")
        else:
            st.error("No BibTeX generated.")
    else:
        # CSV
        if not csv_rows:
            st.error("No CSV data.")
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

st.caption("Note: OpenAI parsing uses the key stored in Streamlit secrets under OPENAI_API_KEY. If you want improved parsing accuracy, add a robust model key. Cross-check uses authors+year+journal (weighted) and the threshold slider controls auto-accept behavior.")
