# streamlit_ref_tool_final.py
"""
Reference → DOI → RIS/BibTeX Streamlit app (final)
Features:
 - Input: Paste references or upload PDF(s)
 - Search order: Crossref -> PubMed (full, then title) -> Semantic Scholar
 - Hybrid AI Parser (Mode B) for offline parsing when search fails or when search result does not match input
 - Auto cross-check: if search result similarity < threshold -> use raw->RIS automatically (can override)
 - Deduplication, editing, export RIS/BibTeX/CSV
"""

import re
import time
import hashlib
import difflib
import csv
import io
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader

# ----------------------------
# Utilities
# ----------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def similarity(a: str, b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# ----------------------------
# PDF extraction
# ----------------------------
def extract_text_from_pdf_file(uploaded) -> str:
    try:
        reader = PdfReader(uploaded)
        pages = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {e}"

# ----------------------------
# Clean & split references
# ----------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # fix hyphenation across line breaks: "adhe\n-sives" -> adhesives
    text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)
    # join single newlines inside paragraphs to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # collapse multiple spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    if not text:
        return []
    text = clean_and_join_broken_lines(text)

    # If explicit numeric markers present, split by them
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            return parts

    # Heuristic split on ". " followed by capital letter or bracket
    cand = re.split(r"\.\s+(?=[A-Z\[])", text)
    results = []
    for c in cand:
        c = c.strip()
        if not c:
            continue
        if not c.endswith("."):
            c = c + "."
        if len(c) < 30 and results:
            results[-1] = results[-1].rstrip(".") + " " + c
        else:
            results.append(c)
    # merge fragments that are unlikely complete references
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    return final

# ----------------------------
# Hybrid AI Parser (Mode B)
# ----------------------------
def split_title_and_journal(ref_text: str) -> Tuple[str, str]:
    """
    Robustly split an inline string where title and journal appear
    Example: "Horsley’s wax. J Perioper Pract. 2007;17:82–84."
      -> ("Horsley’s wax", "J Perioper Pract")
    """
    text = " ".join(ref_text.split())
    # find year to define boundary
    year_match = re.search(r"(19|20)\d{2}", text)
    if year_match:
        year_pos = year_match.start()
        left = text[:year_pos].strip()
    else:
        left = text

    # split left by period/ dot into parts
    parts = [p.strip() for p in left.split(".") if p.strip()]
    if len(parts) == 0:
        return "", ""
    if len(parts) == 1:
        # nothing to separate; return part as title, journal unknown
        return parts[0], ""
    # else first part is likely title, remainder journal pieces
    title = parts[0]
    journal = " ".join(parts[1:])
    # Remove trailing numeric fragments from journal (like volume accidentally captured)
    journal = re.sub(r"\b\d+.*$", "", journal).strip()
    return title.strip().rstrip("."), journal.strip().rstrip(".")

def hybrid_parse_reference(ref: str) -> Dict[str, Any]:
    """
    Hybrid parser combining regex heuristics + small patterns to extract:
    authors (list of strings), year, title, journal, volume, issue, pages, doi
    """
    parsed = {"authors": [], "year": None, "title": "", "journal": "", "volume": "", "issue": "", "pages": "", "doi": None}
    text = clean_and_join_broken_lines(ref)

    # DOI (if present)
    doi_m = DOI_RE.search(text)
    if doi_m:
        parsed["doi"] = doi_m.group(0).rstrip(".,;")

    # Year
    year_m = re.search(r"\((19|20)\d{2}\)|\b(19|20)\d{2}\b", text)
    if year_m:
        parsed["year"] = year_m.group(0).strip("()")

    # Try to isolate authors: often before the first period if it contains names/commas
    first_period_split = re.split(r"\.\s+", text, maxsplit=1)
    if len(first_period_split) >= 2:
        maybe_authors = first_period_split[0].strip()
        remainder = first_period_split[1].strip()
        # Heuristics to determine if the first chunk is authors:
        if ("," in maybe_authors) or re.search(r"\b[A-Z][a-z]{1,}\s+[A-Z]\.?", maybe_authors) or len(maybe_authors.split())<=6:
            # split authors by semicolon or comma between names
            authors_raw = re.split(r";\s*|\.\s*|\s{2,}|,\s*(?=[A-Z][a-z])", maybe_authors)
            # fallback comma split
            if len(authors_raw) == 1:
                authors_raw = re.split(r",\s*", maybe_authors)
            authors = [a.strip().rstrip(".") for a in authors_raw if a.strip()]
            parsed["authors"] = authors
            # For title/journal separation, try split_title_and_journal on remainder
            title_guess, journal_guess = split_title_and_journal(remainder)
            if title_guess:
                parsed["title"] = title_guess
            if journal_guess:
                parsed["journal"] = journal_guess
        else:
            # can't confidently parse authors; try to find quoted title or pattern
            q = re.search(r'“([^”]+)”|\"([^\"]+)\"|\'([^\']+)\'', text)
            if q:
                parsed["title"] = next(g for g in q.groups() if g)
            else:
                # Use split_title_and_journal on whole text
                title_guess, journal_guess = split_title_and_journal(text)
                parsed["title"] = title_guess or parsed["title"]
                parsed["journal"] = journal_guess or parsed["journal"]
    else:
        # Single segment; try to extract quotes or title/journal pattern
        q = re.search(r'“([^”]+)”|\"([^\"]+)\"', text)
        if q:
            parsed["title"] = next(g for g in q.groups() if g)
        else:
            t,j = split_title_and_journal(text)
            parsed["title"] = t
            parsed["journal"] = j

    # Volume / issue / pages extraction (common Vancouver patterns)
    m = re.search(r"(?P<year>(19|20)\d{2})\s*;\s*(?P<vol>\d+)(?:\((?P<iss>\d+)\))?\s*:\s*(?P<pages>[\d\-–]+)", text)
    if m:
        parsed["year"] = parsed["year"] or m.group("year")
        parsed["volume"] = m.group("vol") or ""
        parsed["issue"] = m.group("iss") or ""
        parsed["pages"] = m.group("pages") or ""
        # if journal not detected, attempt to find journal before the year
        if not parsed["journal"]:
            jmatch = re.search(r"([A-Za-z\.\s&\-:]{3,}?)\.\s*" + re.escape(m.group(0)))
            if jmatch:
                parsed["journal"] = jmatch.group(1).strip().rstrip(".")
    else:
        # alternate pattern: Journal. YEAR;VOL:PGS
        m2 = re.search(r"(?P<journal>[A-Za-z\.\s&\-:]{3,}?)\.\s*(?P<year>(19|20)\d{2})\s*;\s*(?P<vol>\d+)\s*:\s*(?P<pages>[\d\-–]+)", text)
        if m2:
            parsed["journal"] = parsed["journal"] or m2.group("journal").strip().rstrip(".")
            parsed["year"] = parsed["year"] or m2.group("year")
            parsed["volume"] = m2.group("vol") or ""
            parsed["pages"] = m2.group("pages") or ""

    # Also attempt to parse pages in simple patterns
    page_m = re.search(r":\s*(\d+[\-–]\d+|\d+)\b", text)
    if page_m and not parsed["pages"]:
        parsed["pages"] = page_m.group(1)

    # Final cleanup: strip whitespace
    parsed["title"] = parsed["title"].strip()
    parsed["journal"] = parsed["journal"].strip()
    parsed["pages"] = parsed["pages"].strip()
    parsed["volume"] = parsed["volume"].strip()
    parsed["issue"] = parsed["issue"].strip()
    parsed["authors"] = [a.strip() for a in parsed["authors"] if a.strip()]
    return parsed

# ----------------------------
# Crossref search (title-based / DOI direct)
# ----------------------------
def crossref_search(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    # direct DOI first
    try:
        doi_m = DOI_RE.search(ref_text)
        if doi_m:
            doi = doi_m.group(0).rstrip(".,;")
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                item = r.json().get("message")
                return item, doi
    except Exception:
        pass

    # build title-like query
    try:
        split_on = re.split(r"\b(19|20)\d{2}\b", ref_text)
        title_guess = split_on[0] if split_on else ref_text
        title_guess = re.sub(r"\bvol\.?.*$", "", title_guess, flags=re.I).strip()
        if not title_guess:
            title_guess = ref_text[:200]
        params = {"query.title": title_guess[:240], "rows": 1}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=12)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if items:
            return items[0], items[0].get("DOI")
    except Exception:
        pass
    return None, None

# ----------------------------
# PubMed search (two-pass)
# ----------------------------
def pubmed_search_full(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db":"pubmed", "term": ref_text, "retmode":"json", "retmax":1}
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None, None
        pmid = ids[0]
        return pubmed_fetch(pmid)
    except Exception:
        return None, None

def pubmed_search_title_only(title: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db":"pubmed", "term": title, "retmode":"json", "retmax":1}
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None, None
        pmid = ids[0]
        return pubmed_fetch(pmid)
    except Exception:
        return None, None

def pubmed_fetch(pmid: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r = requests.get(fetch_url, params={"db":"pubmed", "id":pmid, "retmode":"xml"}, timeout=12)
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
        vol_m = re.search(r"<Volume>(.*?)</Volume>", xml)
        issue_m = re.search(r"<Issue>(.*?)</Issue>", xml)
        doi_m = re.search(r'<ArticleId IdType="doi">(.+?)</ArticleId>', xml)
        vol = vol_m.group(1).strip() if vol_m else ""
        issue = issue_m.group(1).strip() if issue_m else ""
        doi = doi_m.group(1).strip() if doi_m else ""
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

# ----------------------------
# Semantic Scholar fallback
# ----------------------------
def semantic_scholar_search(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        q = ref_text[:300]
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": q, "limit": 1, "fields": "title,authors,year,externalIds,venue"}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data", [])
        if not papers:
            return None, None
        p = papers[0]
        ext = p.get("externalIds") or {}
        doi = ext.get("DOI") or ext.get("DOI:")
        item = {
            "title":[p.get("title","")],
            "author":[{"family": a.get("name","").split()[-1], "given": " ".join(a.get("name","").split()[:-1])} for a in p.get("authors",[])],
            "issued":{"date-parts":[[p.get("year") or 0]]},
            "container-title":[p.get("venue","")],
            "DOI": doi or ""
        }
        return item, doi or None
    except Exception:
        return None, None

# ----------------------------
# Converters: RIS / BibTeX / CSV row builder
# ----------------------------
def convert_item_to_ris(item: Dict[str, Any]) -> str:
    if not item:
        return ""
    lines = []
    typemap = {"journal-article":"JOUR", "book":"BOOK", "book-chapter":"CHAP"}
    ty = typemap.get(item.get("type",""), "GEN")
    lines.append(f"TY  - {ty}")
    if item.get("title"):
        lines.append(f"TI  - {item['title'][0] if isinstance(item['title'], list) else item['title']}")
    for a in item.get("author", [])[:50]:
        if isinstance(a, dict):
            fam = a.get("family","") or ""
            giv = a.get("given","") or ""
            lines.append(f"AU  - {fam}, {giv}")
        else:
            lines.append(f"AU  - {a}")
    if item.get("container-title"):
        ct = item['container-title'][0] if isinstance(item['container-title'], list) else item['container-title']
        lines.append(f"JO  - {ct}")
    if item.get("volume"):
        lines.append(f"VL  - {item.get('volume')}")
    if item.get("issue"):
        lines.append(f"IS  - {item.get('issue')}")
    if item.get("page"):
        p = item["page"]
        if "-" in p:
            sp, ep = p.split("-", 1)
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
    first_author = (item.get("author") or [{}])[0]
    fam = first_author.get("family","") if isinstance(first_author, dict) else str(first_author)
    year = ""
    if item.get("issued",{}).get("date-parts"):
        year = str(item["issued"]["date-parts"][0][0])
    key = re.sub(r"\W+","", fam + year) or "ref"
    btype = "article"
    authors = " and ".join([ (f"{a.get('family','')}, {a.get('given','')}" if isinstance(a, dict) else str(a)) for a in (item.get("author") or []) ])
    title = item.get("title", [""])[0] if item.get("title") else ""
    journal = item.get("container-title", [""])[0] if item.get("container-title") else ""
    doi = item.get("DOI","")
    bib = f"@{btype}{{{key},\n"
    if authors: bib += f"  author = {{{authors}}},\n"
    if title: bib += f"  title = {{{title}}},\n"
    if journal: bib += f"  journal = {{{journal}}},\n"
    if year: bib += f"  year = {{{year}}},\n"
    if doi: bib += f"  doi = {{{doi}}},\n"
    bib = bib.rstrip(",\n") + "\n}\n\n"
    return bib

def parsed_to_item_like(parsed: Dict[str, Any]) -> Dict[str, Any]:
    # create item-like dict for converter uniformity
    item = {
        "title": [parsed.get("title","")],
        "author": [],
        "container-title": [parsed.get("journal","")],
        "issued": {"date-parts": [[int(parsed["year"])]]} if parsed.get("year") and str(parsed.get("year")).isdigit() else {"date-parts": [[0]]},
        "volume": parsed.get("volume",""),
        "issue": parsed.get("issue",""),
        "page": parsed.get("pages",""),
        "DOI": parsed.get("doi","")
    }
    for a in parsed.get("authors", [])[:50]:
        # split author into family/given heuristically
        if "," in a:
            fam,giv = [p.strip() for p in a.split(",",1)]
        else:
            toks = a.split()
            if len(toks) == 1:
                fam = toks[0]; giv = ""
            else:
                fam = toks[-1]; giv = " ".join(toks[:-1])
        item["author"].append({"family": fam, "given": giv})
    return item

def parsed_to_ris(parsed: Dict[str, Any]) -> str:
    item = parsed_to_item_like(parsed)
    return convert_item_to_ris(item)

# ----------------------------
# Deduplication key
# ----------------------------
def canonicalize_for_dedupe(item: Optional[Dict[str,Any]], ref_text: str) -> str:
    if item and item.get("DOI"):
        return item["DOI"].lower()
    s = re.sub(r"\W+", " ", ref_text.lower())
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Reference → RIS (Final)", layout="wide")
st.title("📚 Reference → DOI → RIS — Final (Crossref + PubMed + Semantic Scholar + Hybrid AI Parser)")

st.markdown("""
Paste references (or upload PDFs). For each reference the app will:
1. Try Crossref → PubMed (full, then title) → Semantic Scholar.
2. Cross-check found metadata with your pasted reference (similarity).
   - If similarity >= threshold → found metadata will be preselected.
   - If similarity < threshold → the app will default to converting your pasted reference to RIS (you may still override).
3. You can edit results and export RIS / BibTeX / CSV.
""")

# Input mode
mode = st.radio("Input method", ["Paste references", "Upload PDF(s)"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references here (supports numbered forms like [1], 1., 1) etc.)", height=350)
else:
    uploaded = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        parts = []
        for f in uploaded:
            with st.spinner(f"Extracting {f.name}..."):
                txt = extract_text_from_pdf_file(f)
                if txt.startswith("ERROR_PDF_EXTRACT"):
                    st.error(f"Error extracting {f.name}: {txt}")
                else:
                    # try to locate References section
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                    block = m.group(2) if m else txt
                    parts.append(block)
        raw_text = "\n\n".join(parts)
        if raw_text:
            st.text_area("Extracted text from PDF", raw_text, height=200)

auto_accept = st.checkbox("Auto-accept search result when similarity >= threshold", value=True)
threshold = st.slider("Auto-accept similarity threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.01)
export_format = st.selectbox("Export format", ["RIS", "BibTeX", "CSV"], index=0)

if st.button("Process & Generate"):
    if not raw_text.strip():
        st.warning("Please paste references or upload PDFs first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected after splitting.")
        st.stop()
    st.success(f"Detected {len(refs)} reference(s).")

    # Search & parse
    results = []
    seen_keys = set()
    progress = st.progress(0)
    status = st.empty()

    for i, ref in enumerate(refs, start=1):
        status.text(f"Searching reference {i}/{len(refs)}")
        # initial search attempts
        item = None; doi = None; source = None

        # Crossref
        cr_item, cr_doi = crossref_search(ref)
        if cr_item and cr_doi:
            item, doi = cr_item, cr_doi
            source = "Crossref"

        # PubMed (full then title)
        if not doi:
            pm_item, pm_doi = pubmed_search_full(ref)
            if pm_item and pm_doi:
                item, doi = pm_item, pm_doi
                source = "PubMed (full)"
            else:
                parsed_guess = hybrid_parse_reference(ref)
                title_guess = parsed_guess.get("title") or ""
                if title_guess:
                    pm2_item, pm2_doi = pubmed_search_title_only(title_guess)
                    if pm2_item and pm2_doi:
                        item, doi = pm2_item, pm2_doi
                        source = "PubMed (title)"

        # Semantic Scholar fallback
        if not doi:
            ss_item, ss_doi = semantic_scholar_search(ref)
            if ss_item:
                item, doi = ss_item, ss_doi
                source = "SemanticScholar"

        # compute similarity between pasted ref and found title
        found_title = item.get("title",[None])[0] if item and item.get("title") else None
        sim_score = similarity(ref, found_title) if found_title else 0.0

        # hybrid parsed (always compute for fallback)
        parsed = hybrid_parse_reference(ref)

        # dedupe key
        key = canonicalize_for_dedupe(item, ref)
        duplicate = key in seen_keys
        if not duplicate:
            seen_keys.add(key)

        results.append({
            "original": ref,
            "item": item,
            "doi": doi,
            "source": source,
            "similarity": sim_score,
            "parsed": parsed,
            "duplicate": duplicate
        })
        progress.progress(i/len(refs))
        time.sleep(0.12)

    status.empty()
    progress.empty()

    # Interactive Review: default selection respects auto-cross-check rule
    st.header("Review detected references")
    st.write("If a search result was found but similarity < threshold, the app defaults to converting the pasted reference to RIS. You can still override and accept the search metadata for any reference.")

    chosen = []  # list of dicts {include, use_search, final_item, final_parsed, source}
    for idx, r in enumerate(results, start=1):
        with st.expander(f"Reference {idx}: {r['original'][:200]}{'...' if len(r['original'])>200 else ''}"):
            if r["duplicate"]:
                st.info("Duplicate detected — you may skip or still include.")
            if r["item"]:
                st.markdown(f"**Found ({r['source']}) — similarity {r['similarity']:.2f}**")
                title_display = r["item"].get("title",[r['parsed'].get("title","")])[0] if r.get("item") else r['parsed'].get("title","")
                st.write("Title (found):", title_display)
                if r.get("doi"):
                    st.write("DOI:", r["doi"])
                # show authors preview
                if r["item"] and r["item"].get("author"):
                    auths = r["item"].get("author")
                    preview = []
                    for a in auths[:6]:
                        if isinstance(a, dict):
                            preview.append(f"{a.get('family','')} {a.get('given','')}")
                        else:
                            preview.append(str(a))
                    st.write("Authors (preview):", ", ".join(preview))
            else:
                st.warning("No search metadata found.")

            # Show parsed preview always
            st.markdown("**Hybrid-parsed (auto)**")
            parsed = r.get("parsed", {})
            st.write(f"Authors: {parsed.get('authors')}")
            st.write(f"Title: {parsed.get('title')}")
            st.write(f"Journal: {parsed.get('journal')}")
            st.write(f"Year: {parsed.get('year')}")
            st.write(f"Volume/Issue/Pages: {parsed.get('volume')}/{parsed.get('issue')}/{parsed.get('pages')}")
            if parsed.get("doi"):
                st.write("DOI (in text):", parsed.get("doi"))

            # Default selection logic:
            # - If item found AND similarity >= threshold => default use_search True
            # - If item found AND similarity < threshold => default use_search False (auto-convert raw)
            default_use_search = False
            if r["item"] and auto_accept and r["similarity"] >= threshold:
                default_use_search = True
            elif r["item"] and r["similarity"] < threshold:
                default_use_search = False

            choice = st.radio(
                f"Choose action for ref {idx}",
                options=["Use found metadata (if any)", "Use pasted reference → Hybrid parser → RIS"],
                index=0 if default_use_search else 1,
                key=f"choice_{idx}"
            )
            include = st.checkbox(f"Include this reference in export (Ref {idx})", value=True, key=f"include_{idx}")

            chosen.append({
                "include": include,
                "use_search": (choice == "Use found metadata (if any)"),
                "result": r
            })

    # Build final outputs according to choices
    ris_blocks = []
    bib_blocks = []
    csv_rows = []
    for ent in chosen:
        if not ent["include"]:
            continue
        r = ent["result"]
        if ent["use_search"] and r["item"]:
            # Validate cross-check again: if similarity < threshold, app auto-converted earlier by default.
            # But if user explicitly chose use_search, we accept it.
            ris_blocks.append(convert_item_to_ris(r["item"]))
            bib_blocks.append(convert_item_to_bibtex(r["item"]))
            # CSV row
            csv_rows.append({
                "original": r["original"],
                "title": r["item"].get("title",[r["parsed"].get("title","")])[0] if r["item"].get("title") else r["parsed"].get("title",""),
                "doi": r.get("doi") or "",
                "journal": (r["item"].get("container-title",[r["parsed"].get("journal","")])[0] if r["item"].get("container-title") else r["parsed"].get("journal","")),
                "year": r["item"].get("issued",{}).get("date-parts",[[r["parsed"].get("year","")]])[0][0] if r["item"].get("issued") else r["parsed"].get("year",""),
                "source": r.get("source") or "parsed"
            })
        else:
            # Use hybrid parsed to produce RIS
            parsed = r["parsed"]
            ris_blocks.append(parsed_to_ris(parsed))
            # Build item-like for bib conversion
            item_like = parsed_to_item_like(parsed)
            bib_blocks.append(convert_item_to_bibtex(item_like))
            csv_rows.append({
                "original": r["original"],
                "title": parsed.get("title",""),
                "doi": parsed.get("doi","") or "",
                "journal": parsed.get("journal",""),
                "year": parsed.get("year",""),
                "source": "parsed"
            })

    # Final export UI
    st.header("Export")
    total_entries = len(ris_blocks)
    st.success(f"Prepared {total_entries} entries for export.")

    if export_format == "RIS":
        final_ris = "".join(ris_blocks)
        if not final_ris.strip():
            st.error("No RIS output generated.")
        else:
            st.download_button("Download RIS file", data=final_ris, file_name="references.ris", mime="application/x-research-info-systems")
            with st.expander("RIS preview"):
                st.code(final_ris, language="text")
    elif export_format == "BibTeX":
        final_bib = "".join(bib_blocks)
        if not final_bib.strip():
            st.error("No BibTeX output generated.")
        else:
            st.download_button("Download BibTeX file", data=final_bib, file_name="references.bib", mime="text/x-bibtex")
            with st.expander("BibTeX preview"):
                st.code(final_bib, language="text")
    else:  # CSV
        # produce CSV from csv_rows
        if not csv_rows:
            st.error("No CSV data generated.")
        else:
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=["original","title","doi","journal","year","source"])
            writer.writeheader()
            for row in csv_rows:
                writer.writerow(row)
            csv_data = csv_buf.getvalue()
            st.download_button("Download CSV", data=csv_data, file_name="references.csv", mime="text/csv")
            with st.expander("CSV preview"):
                st.code(csv_data, language="text")

st.caption("Search order: Crossref → PubMed (full then title) → Semantic Scholar. If search result similarity < threshold, default is to convert pasted reference to RIS. You can override per-reference.")
