# streamlit_ref_tool_hybrid.py
"""
Reference → DOI → RIS/BibTeX Streamlit app with Hybrid AI Parser (Mode B)
Search order: Crossref (title) -> PubMed (full ref then title) -> Semantic Scholar (fallback)
If searches fail or user chooses, a Hybrid Parser extracts fields and constructs RIS entries.
Requirements:
    streamlit
    pypdf
    requests
Save as streamlit_ref_tool_hybrid.py and run with:
    streamlit run streamlit_ref_tool_hybrid.py
"""
import re
import time
import hashlib
import difflib
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader

# -------------------------
# Utilities
# -------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def similarity(a: str, b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# -------------------------
# PDF extraction
# -------------------------
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

# -------------------------
# Clean & split references
# -------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)  # fix hyphenation
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)    # join single-line breaks
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    if not text:
        return []
    text = clean_and_join_broken_lines(text)

    # If numeric markers present like [1], 1., 1)
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            return parts

    # Heuristic split on ". " followed by uppercase or bracket
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
    # Merge fragments without year into previous
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    return final

# -------------------------
# Hybrid AI Parser (Mode B) - heuristics + pattern features
# -------------------------
def hybrid_parse_reference(ref: str) -> Dict[str, Any]:
    """
    Hybrid parser combining structured regex heuristics + pattern detection.
    Returns a dict with fields:
    authors (list of strings), year, title, journal, volume, issue, pages, doi
    """
    parsed = {"authors": [], "year": None, "title": "", "journal": "", "volume": "", "issue": "", "pages": "", "doi": None}

    text = ref.strip()

    # 1) DOI
    doi_m = DOI_RE.search(text)
    if doi_m:
        parsed["doi"] = doi_m.group(0).rstrip(".,;")

    # 2) Year (look for (2005) or 2005; or 2005.)
    year_m = re.search(r"\((19|20)\d{2}\)", text) or re.search(r"\b(19|20)\d{2}\b", text)
    if year_m:
        parsed["year"] = year_m.group(0).strip("()")

    # 3) Try to split authors and remainder
    # Authors usually appear before the first period that precedes the title
    parts = re.split(r"\.\s+", text, maxsplit=1)
    if len(parts) >= 2:
        maybe_authors = parts[0].strip()
        remainder = parts[1].strip()
        # If first segment contains comma-separated names (common), we treat as authors
        if re.search(r"[A-Za-z\-']+,\s*[A-Za-z]", maybe_authors) or re.search(r"[A-Z]\.\s*[A-Z]", maybe_authors) or len(maybe_authors.split()) <= 6:
            # split by semicolon or comma with caution
            authors = re.split(r";\s*|\s{2,}|,\s*(?=[A-Z][a-z])", maybe_authors)
            # fallback simpler split by comma when initials present
            if len(authors) == 1:
                authors = re.split(r",\s*", maybe_authors)
            authors = [a.strip().rstrip(".") for a in authors if a.strip()]
            parsed["authors"] = authors

            # 4) Title detection - heuristic: title often ends before journal or before year/volume markers
            title_guess = remainder
            # cut off at journal-like patterns: "JournalName. 2005;" or "JournalName 2005;"
            # find typical volume/page markers like "2005;17(3):811-14" or "17(3):811-14"
            volpage = re.search(r"(\d{4}\s*;\s*\d+|\d+\s*\(\d+\)|:\s*\d{1,4}[\-–]\d{1,4}|\b\d+:\d{1,4})", remainder)
            if volpage:
                title_guess = remainder.split(volpage.group(0))[0].strip()
            # also cut if we see pattern ". J " or ". The " where a journal name starts
            journal_start = re.search(r"\.\s+[A-Z][A-Za-z]+\s", remainder)
            if journal_start:
                title_guess = remainder.split(journal_start.group(0))[0].strip()
            # if quotes present, title is probably in quotes
            q = re.search(r'“([^”]+)”|\"([^\"]+)\"|‘([^’]+)’', remainder)
            if q:
                title_str = next(g for g in q.groups() if g)
                parsed["title"] = title_str.strip().rstrip(".")
            else:
                parsed["title"] = title_guess.strip().rstrip(".")
        else:
            # cannot clearly detect authors -> attempt to find title in quotes or before journal-like pattern
            q = re.search(r'“([^”]+)”|\"([^\"]+)\"', text)
            if q:
                parsed["title"] = next(g for g in q.groups() if g).strip().rstrip(".")
            else:
                # fallback: take the first sentence as title
                parsed["title"] = parts[0].strip().rstrip(".")
    else:
        # no dot-split found
        q = re.search(r'“([^”]+)”|\"([^\"]+)\"', text)
        if q:
            parsed["title"] = next(g for g in q.groups() if g).strip().rstrip(".")
        else:
            # try to find pattern Title. Journal Year;Volume:Pages
            m = re.search(r"^(.*?)(?:\.\s+)?([A-Z][A-Za-z\.\s&-]{3,})\.\s*(?:19|20)\d{2}", text)
            if m:
                parsed["title"] = m.group(1).strip().rstrip(".")
                parsed["journal"] = m.group(2).strip().rstrip(".")
            else:
                parsed["title"] = text[:200].strip()

    # 5) Journal, volume, issue, pages extraction with several common patterns
    # Pattern: "Journal Name. 2005;17(3):811-21."
    m = re.search(r"(?P<journal>[A-Za-z\.\s&\-:]{3,}?)\.\s*(?P<year>(19|20)\d{2})\s*;\s*(?P<vol>\d+)(?:\((?P<iss>\d+)\))?\s*:\s*(?P<pages>[\d\-–]+)", text)
    if m:
        parsed["journal"] = parsed["journal"] or m.group("journal").strip().rstrip(".")
        parsed["year"] = parsed["year"] or m.group("year")
        parsed["volume"] = m.group("vol") or ""
        parsed["issue"] = m.group("iss") or ""
        parsed["pages"] = m.group("pages") or ""
    else:
        # pattern: "Journal. 2005;16:497–514."
        m2 = re.search(r"(?P<journal>[A-Za-z\.\s&\-:]{3,}?)\.\s*(?P<year>(19|20)\d{2})\s*;\s*(?P<vol>\d+)\s*:\s*(?P<pages>[\d\-–]+)", text)
        if m2:
            parsed["journal"] = parsed["journal"] or m2.group("journal").strip().rstrip(".")
            parsed["year"] = parsed["year"] or m2.group("year")
            parsed["volume"] = m2.group("vol") or ""
            parsed["pages"] = m2.group("pages") or ""

    # pattern: "J Oral Pathol. 1984;13(6):661-70."
    m3 = re.search(r"(?P<journal>[A-Za-z\.\s&\-:]{2,}?)\.\s*(?P<year>(19|20)\d{2})\s*;\s*(?P<vol>\d+)\((?P<iss>\d+)\)\s*:\s*(?P<pages>[\d\-–]+)", text)
    if m3:
        parsed["journal"] = parsed["journal"] or m3.group("journal").strip().rstrip(".")
        parsed["year"] = parsed["year"] or m3.group("year")
        parsed["volume"] = m3.group("vol")
        parsed["issue"] = m3.group("iss")
        parsed["pages"] = m3.group("pages")

    # pattern: "Volume (Issue): pages" alone
    volonly = re.search(r"(?P<vol>\d+)\s*\(?(\s*(?P<iss>\d+)\s*)?\)?\s*:\s*(?P<pages>[\d\-–]+)", text)
    if volonly and not parsed["volume"]:
        parsed["volume"] = volonly.group("vol")
        parsed["issue"] = volonly.group("iss") or ""
        parsed["pages"] = volonly.group("pages") or parsed["pages"]

    # Clean fields
    parsed["title"] = parsed["title"].strip()
    parsed["journal"] = parsed["journal"].strip()
    parsed["pages"] = parsed["pages"].strip()
    parsed["volume"] = parsed["volume"].strip()
    parsed["issue"] = parsed["issue"].strip()
    if isinstance(parsed["authors"], list):
        parsed["authors"] = [a.strip() for a in parsed["authors"] if a.strip()]
    return parsed

# -------------------------
# Crossref search (title based / DOI direct)
# -------------------------
def crossref_search(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    # try DOI direct
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

# -------------------------
# PubMed search: two-pass (full ref then extracted title)
# -------------------------
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

# -------------------------
# Semantic Scholar fallback
# -------------------------
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

# -------------------------
# Converters: RIS & BibTeX
# -------------------------
def convert_item_to_ris(item: Dict[str, Any]) -> str:
    if not item:
        return ""
    lines = []
    typemap = {"journal-article":"JOUR", "book":"BOOK", "book-chapter":"CHAP"}
    ty = typemap.get(item.get("type",""), "GEN")
    lines.append(f"TY  - {ty}")
    if item.get("title"):
        lines.append(f"TI  - {item['title'][0]}")
    for a in item.get("author", [])[:50]:
        fam = a.get("family","") or ""
        giv = a.get("given","") or ""
        lines.append(f"AU  - {fam}, {giv}")
    if item.get("container-title"):
        lines.append(f"JO  - {item['container-title'][0]}")
    if item.get("volume"):
        lines.append(f"VL  - {item['volume']}")
    if item.get("issue"):
        lines.append(f"IS  - {item['issue']}")
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
    first_author = (item.get("author") or [{}])[0].get("family","ref")
    year = ""
    if item.get("issued",{}).get("date-parts"):
        year = str(item["issued"]["date-parts"][0][0])
    key = re.sub(r"\W+","", first_author + year) or "ref"
    btype = "article"
    authors = " and ".join([f"{a.get('family','')}, {a.get('given','')}" for a in (item.get("author") or [])])
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

def parsed_raw_to_ris(parsed: Dict[str, Any]) -> str:
    lines = []
    lines.append("TY  - GEN")
    if parsed.get("title"):
        lines.append(f"TI  - {parsed['title']}")
    for a in parsed.get("authors", [])[:50]:
        lines.append(f"AU  - {a}")
    if parsed.get("journal"):
        lines.append(f"JO  - {parsed['journal']}")
    if parsed.get("volume"):
        lines.append(f"VL  - {parsed['volume']}")
    if parsed.get("issue"):
        lines.append(f"IS  - {parsed['issue']}")
    if parsed.get("pages"):
        pg = parsed.get("pages")
        if "-" in pg:
            sp, ep = pg.split("-",1)
            lines.append(f"SP  - {sp}")
            lines.append(f"EP  - {ep}")
        else:
            lines.append(f"SP  - {pg}")
    if parsed.get("year"):
        lines.append(f"PY  - {parsed['year']}")
    if parsed.get("doi"):
        lines.append(f"DO  - {parsed['doi']}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

# -------------------------
# Deduplication key
# -------------------------
def canonicalize_for_dedupe(item: Optional[Dict[str,Any]], ref_text: str) -> str:
    if item and item.get("DOI"):
        return item["DOI"].lower()
    s = re.sub(r"\W+", " ", ref_text.lower())
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="Reference → RIS (Hybrid Parser)", layout="wide")
st.title("📚 Reference → DOI → RIS (Crossref + PubMed + Semantic Scholar + Hybrid AI Parser)")
st.write("Paste references or upload PDF. The app searches metadata and — when you choose — uses a Hybrid AI Parser to extract fields and create RIS entries.")

mode = st.radio("Input method", ["Paste references", "Upload PDF"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references here (supports [1], 1., 1) etc.)", height=300)
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
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                    block = m.group(2) if m else txt
                    parts.append(block)
        raw_text = "\n\n".join(parts)
        if raw_text:
            st.text_area("Extracted text", raw_text, height=200)

auto_accept = st.checkbox("Auto-accept search results when similarity >= threshold", value=True)
threshold = st.slider("Auto-accept similarity threshold", min_value=0.0, max_value=1.0, value=0.65, step=0.05)
export_format = st.selectbox("Export format", ["RIS", "BibTeX"], index=0)

if st.button("Process references"):
    if not raw_text.strip():
        st.warning("Paste references or upload a PDF first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected.")
        st.stop()

    st.success(f"Detected {len(refs)} references.")
    st.info("Searching Crossref → PubMed (full ref then title) → Semantic Scholar. If you prefer, use Hybrid Parser to convert raw reference to RIS.")

    all_results = []
    seen_keys = set()
    ris_blocks = []
    bib_blocks = []

    progress = st.progress(0)
    status = st.empty()

    # Search & parse loop
    for i, ref in enumerate(refs, start=1):
        status.text(f"Processing reference {i}/{len(refs)}")
        item = None
        doi = None
        source = None

        # Crossref
        cr_item, cr_doi = crossref_search(ref)
        if cr_item and cr_doi:
            item, doi = cr_item, cr_doi
            source = "Crossref"
        else:
            # PubMed full text
            pm_item, pm_doi = pubmed_search_full(ref)
            if pm_item and pm_doi:
                item, doi = pm_item, pm_doi
                source = "PubMed (full)"
            else:
                # PubMed title only (use parsed title)
                parsed = hybrid_parse_reference(ref)
                title_guess = parsed.get("title") or ""
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

        found_title = item.get("title",[None])[0] if item else None
        sim_score = similarity(ref, found_title) if found_title else 0.0

        key = canonicalize_for_dedupe(item, ref)
        duplicate = key in seen_keys
        if not duplicate:
            seen_keys.add(key)

        all_results.append({
            "original": ref,
            "item": item,
            "doi": doi,
            "source": source,
            "sim": sim_score,
            "duplicate": duplicate,
            "parsed": hybrid_parse_reference(ref)  # precompute hybrid parse for quick choice
        })
        progress.progress(i/len(refs))
        time.sleep(0.12)

    status.empty()
    progress.empty()

    # Interactive review
    st.header("Review & choose per reference")
    user_choices = []
    for idx, r in enumerate(all_results, start=1):
        with st.expander(f"Reference {idx} — {r['original'][:200]}{'...' if len(r['original'])>200 else ''}"):
            if r["duplicate"]:
                st.info("Duplicate detected — you may skip or still include.")
            if r["item"]:
                st.markdown(f"**Found ({r['source']}) — similarity {r['sim']:.2f}**")
                st.write("Title:", r["item"].get("title", [""])[0] if r["item"].get("title") else "")
                if r.get("doi"):
                    st.write("DOI:", r["doi"])
                if r["item"].get("author"):
                    names_preview = ", ".join([f"{a.get('family','')} {a.get('given','')}".strip() for a in r["item"].get("author",[])][:6])
                    if names_preview:
                        st.write("Authors (preview):", names_preview)
            else:
                st.warning("No search metadata found.")

            # Show hybrid parsed preview
            parsed = r.get("parsed", {})
            st.markdown("**Hybrid parser preview (auto)**")
            st.write(f"Authors: {parsed.get('authors')}")
            st.write(f"Title: {parsed.get('title')}")
            st.write(f"Journal: {parsed.get('journal')}")
            st.write(f"Year: {parsed.get('year')}")
            st.write(f"Volume/Issue/Pages: {parsed.get('volume')}/{parsed.get('issue')}/{parsed.get('pages')}")
            if parsed.get("doi"):
                st.write("DOI (found in text):", parsed.get("doi"))

            default_choice = 1  # 0 -> use search, 1 -> use hybrid raw
            if r["item"] and auto_accept and r["sim"] >= threshold:
                default_choice = 0

            choice = st.radio(
                f"Action for reference {idx}",
                ("Use found metadata (if any)", "Use Hybrid parser → convert raw reference to RIS"),
                index=default_choice,
                key=f"choice_{idx}"
            )
            include_checkbox = st.checkbox(f"Include this reference in export (Ref {idx})", value=True, key=f"include_{idx}")

            user_choices.append({
                "include": include_checkbox,
                "choice": choice,
                "result": r
            })

    # Build outputs
    for entry in user_choices:
        if not entry["include"]:
            continue
        r = entry["result"]
        if entry["choice"] == "Use found metadata (if any)" and r["item"]:
            # use search item
            if export_format == "RIS":
                ris_blocks.append(convert_item_to_ris(r["item"]))
            else:
                bib_blocks.append(convert_item_to_bibtex(r["item"]))
        else:
            # use hybrid parsed version
            parsed = r.get("parsed", {})
            # Build a crossref-like item for uniform conversion
            item_like = {
                "title": [parsed.get("title","")],
                "author": [],
                "container-title": [parsed.get("journal","")],
                "issued": {"date-parts": [[int(parsed.get("year"))]]} if parsed.get("year") and str(parsed.get("year")).isdigit() else {"date-parts": [[0]]},
                "volume": parsed.get("volume",""),
                "issue": parsed.get("issue",""),
                "page": parsed.get("pages",""),
                "DOI": parsed.get("doi","") or ""
            }
            # authors as "family, given" are not parsed perfectly; keep raw strings
            # convert parsed authors into simple author dicts if possible
            authors = []
            for a in parsed.get("authors", [])[:50]:
                # If "Surname Initials" or "Surname, Given" style detection
                if "," in a:
                    parts = [p.strip() for p in a.split(",",1)]
                    family = parts[0]
                    given = parts[1] if len(parts) > 1 else ""
                else:
                    # split last token as family
                    toks = a.split()
                    if len(toks) == 1:
                        family = toks[0]; given = ""
                    else:
                        family = toks[-1]; given = " ".join(toks[:-1])
                authors.append({"family": family, "given": given})
            item_like["author"] = authors

            if export_format == "RIS":
                ris_blocks.append(convert_item_to_ris(item_like))
            else:
                bib_blocks.append(convert_item_to_bibtex(item_like))

    # Finalize outputs and downloads
    total_included = sum(1 for e in user_choices if e["include"])
    st.success(f"Prepared export with {total_included} references.")

    if export_format == "RIS":
        final_ris = "".join(ris_blocks)
        if not final_ris.strip():
            st.error("No RIS content generated.")
        else:
            st.download_button("Download RIS", data=final_ris, file_name="references.ris", mime="application/x-research-info-systems")
            with st.expander("RIS preview"):
                st.code(final_ris, language="text")
    else:
        final_bib = "".join(bib_blocks)
        if not final_bib.strip():
            st.warning("No BibTeX content generated.")
        st.download_button("Download BibTeX", data=final_bib, file_name="references.bib", mime="text/x-bibtex")
        with st.expander("BibTeX preview"):
            st.code(final_bib, language="text")

st.caption("Search order: Crossref → PubMed (full ref then title) → Semantic Scholar. Hybrid parser = Mode B (heuristics + pattern detection). Adjust auto-accept threshold to speed decisions.")
