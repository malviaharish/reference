# streamlit_ref_tool.py
import re
import time
import hashlib
import difflib
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader  # Streamlit Cloud friendly

# -------------------------
#  Utility functions
# -------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip().lower()

def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()

# -------------------------
#  PDF extraction
# -------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    try:
        reader = PdfReader(uploaded_file)
        pages = []
        for p in reader.pages:
            txt = p.extract_text()
            if txt:
                pages.append(txt)
        return "\n".join(pages)
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {e}"

# -------------------------
#  Reference splitting (supports numbered, [1], unnumbered)
# -------------------------
def clean_and_join_broken_lines(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # fix hyphenation across lines
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # merge single-line breaks within paragraphs to spaces
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # collapse multiple whitespace
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    """
    Return list of reference strings.
    Supports: numeric markers (1., 1), [1]) and non-numbered lists.
    Joins broken/wrapped lines.
    """
    if not text:
        return []

    text = clean_and_join_broken_lines(text)

    # If explicit numeric markers exist, split by them
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 15]
        if parts:
            return parts

    # Otherwise heuristic split: split on ". " followed by capital letter (likely next ref)
    cand = re.split(r"\.\s+(?=[A-Z\[])", text)
    results = []
    for c in cand:
        c = c.strip()
        if not c:
            continue
        # append trailing period if missing
        if not c.endswith("."):
            c = c + "."
        # filter out very short junk lines
        if len(c) < 20:
            if results:
                results[-1] = results[-1].rstrip(".") + " " + c
            else:
                results.append(c)
        else:
            results.append(c)
    # Final merge for fragments without year/journal
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    return final

# -------------------------
#  External search: Crossref (title) and Semantic Scholar
# -------------------------
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

def search_crossref_title(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Use Crossref bibliographic/title search.
    Return (item_dict, doi) or (None, None).
    We try to build a reasonable query: extract phrase before year/journal markers.
    """
    try:
        # Try direct DOI if present
        m = DOI_RE.search(ref_text)
        if m:
            doi = m.group(0).rstrip(".,;")
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json().get("message")
                return data, doi

        # Make a title-like query
        # Cut off at year if present
        parts = re.split(r"\b(19|20)\d{2}\b", ref_text)
        title_guess = parts[0] if parts else ref_text
        # also remove trailing journal-like patterns (vol., pp.)
        title_guess = re.sub(r"\bvol\.?\b.*$", "", title_guess, flags=re.I)
        title_guess = re.sub(r"\bpp?\.?\s*\d+.*$", "", title_guess, flags=re.I)
        title_guess = title_guess.strip()
        if not title_guess:
            title_guess = ref_text[:200]

        params = {"query.title": title_guess[:240], "rows": 1}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=12)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None, None
        item = items[0]
        return item, item.get("DOI")
    except Exception:
        return None, None

def search_semantic_scholar(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Use Semantic Scholar free search API.
    Returns (item_dict, doi) or (None, None).
    """
    try:
        q = ref_text[:300]
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {"query": q, "limit": 1, "fields": "title,authors,year,externalIds"}
        r = requests.get(url, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        papers = data.get("data", [])
        if not papers:
            return None, None
        p = papers[0]
        doi = None
        ext = p.get("externalIds", {}) or {}
        # Semantic Scholar returns DOI under 'DOI' key sometimes
        doi = ext.get("DOI") or ext.get("DOI:")
        # Build Crossref-like item shape (partial)
        item = {
            "title": [p.get("title", "")],
            "author": [{"family": a.get("name","").split()[-1], "given": " ".join(a.get("name","").split()[:-1])} for a in p.get("authors", [])],
            "issued": {"date-parts": [[p.get("year") or 0]]},
            "container-title": [""],
            "DOI": doi or ""
        }
        return item, doi or ""
    except Exception:
        return None, None

# -------------------------
#  Converters
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
    for a in item.get("author", []):
        fam = a.get("family", "") or ""
        giv = a.get("given", "") or ""
        lines.append(f"AU  - {fam}, {giv}")
    if item.get("container-title"):
        ct = item.get("container-title")[0] if item.get("container-title") else ""
        lines.append(f"JO  - {ct}")
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
    if item.get("issued", {}).get("date-parts"):
        year = str(item["issued"]["date-parts"][0][0])
    key = re.sub(r"\W+","", first_author + year) or "ref"
    btype = "article"
    authors = " and ".join([f"{a.get('family','')}, {a.get('given','')}" for a in (item.get("author") or [])])
    title = item.get("title", [""])[0]
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

def raw_ref_to_ris(raw: str) -> str:
    txt = raw.strip()
    if len(txt) > 250:
        txt = txt[:247] + "..."
    return f"TY  - GEN\nTI  - {txt}\nER  - \n\n"

# -------------------------
#  Deduplication helper
# -------------------------
def canonicalize_for_dedupe(item: Optional[Dict[str,Any]], ref_text: str) -> str:
    if item and item.get("DOI"):
        return item["DOI"].lower()
    s = ref_text.lower()
    s = re.sub(r"\W+", " ", s)
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -------------------------
#  Streamlit UI
# -------------------------
st.set_page_config(page_title="Reference → RIS (Crossref + Semantic Scholar)", layout="wide")
st.title("📚 Reference → DOI → RIS (Crossref + Semantic Scholar)")
st.write("Paste references or upload PDF. For each detected reference you can accept the search result or convert the raw reference to RIS.")

# Input mode
mode = st.radio("Input method", ["Paste references", "Upload PDF"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references here (supports numbered forms like [1], 1., 1) etc.)", height=300)
else:
    uploaded = st.file_uploader("Upload PDF file(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        pdf_texts = []
        for f in uploaded:
            with st.spinner(f"Extracting {f.name}..."):
                txt = extract_text_from_pdf(f)
                if txt.startswith("ERROR_PDF_EXTRACT"):
                    st.error(f"Error extracting {f.name}: {txt}")
                else:
                    # attempt to find References section heuristically
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{100,200000})", txt)
                    if m:
                        block = m.group(2)
                    else:
                        block = txt
                    pdf_texts.append(block)
        raw_text = "\n\n".join(pdf_texts)
        if raw_text:
            st.text_area("Extracted text", raw_text, height=200)

# Options
auto_accept = st.checkbox("Auto-accept search results when similarity >= threshold", value=True)
threshold = st.slider("Auto-accept similarity threshold", min_value=0.0, max_value=1.0, value=0.6, step=0.05)
export_format = st.selectbox("Export format", ["RIS", "BibTeX"], index=0)

# Process button
if st.button("Split & Search"):
    if not raw_text.strip():
        st.warning("Paste references or upload a PDF first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)

    if not refs:
        st.warning("No references detected after splitting.")
        st.stop()

    st.success(f"Detected {len(refs)} references.")
    st.info("Now searching Crossref (title) → Semantic Scholar fallback. This may take a few seconds per reference.")

    results = []  # each entry: dict with original, item(found), doi, source, sim
    progress = st.progress(0)
    status = st.empty()

    for i, ref in enumerate(refs, start=1):
        status.text(f"Searching for reference {i}/{len(refs)}")
        # prioritized searches
        item, doi = search_crossref_title(ref)
        source = "Crossref" if doi else None

        if not doi:
            item, doi = search_semantic_scholar(ref)
            if doi or item:
                source = "SemanticScholar"

        # prepare display fields
        found_title = item.get("title",[None])[0] if item else None
        sim = similarity(ref, found_title) if found_title else 0.0

        results.append({
            "original": ref,
            "item": item,
            "doi": doi or None,
            "source": source,
            "similarity": sim
        })

        progress.progress(i/len(refs))
        time.sleep(0.15)  # polite pause

    status.empty()
    progress.empty()

    # Deduplicate by DOI or fallback canonicalization
    seen = set()
    filtered = []
    for r in results:
        key = canonicalize_for_dedupe(r.get("item"), r["original"])
        if key in seen:
            r["status"] = "duplicate"
            filtered.append(r)
        else:
            seen.add(key)
            r["status"] = "new"
            filtered.append(r)
    results = filtered

    # Show interactive review where user can choose Accept Search Result vs Use Raw->RIS
    st.header("Review & Choose for each reference")
    st.write("For each detected reference, choose whether to use the found metadata (preferred) or convert the original pasted text to a minimal RIS entry.")
    st.write("If Auto-accept is enabled and similarity >= threshold, the found metadata is pre-selected.")

    user_choices = []  # store tuples (use_search_bool, item/ref)
    for idx, r in enumerate(results, start=1):
        with st.expander(f"Reference {idx} — {r['original'][:140]}{'...' if len(r['original'])>140 else ''}"):
            if r["status"] == "duplicate":
                st.info("This entry appears to be a duplicate (skipped earlier).")
            if r["item"]:
                st.markdown(f"**Found ({r['source']})** — similarity: **{r['similarity']:.2f}**")
                st.write("Title:", r["item"].get("title", [""])[0])
                if r["item"].get("container-title"):
                    st.write("Journal:", r["item"].get("container-title", [""])[0])
                if r.get("doi"):
                    st.write("DOI:", r["doi"])
            else:
                st.warning("No metadata found by Crossref or Semantic Scholar for this reference.")

            # default choice logic
            default_use_search = False
            if r["item"] and auto_accept and r["similarity"] >= threshold:
                default_use_search = True

            choice = st.radio(
                label=f"Choose action for reference {idx}",
                options=["Use found metadata (if any)", "Use raw pasted reference → minimal RIS"],
                index=0 if default_use_search else 1,
                key=f"choice_{idx}"
            )
            use_search = (choice == "Use found metadata (if any)")
            user_choices.append((use_search, r))

    # Build final output based on user choices
    ris_blocks = []
    bib_blocks = []
    final_results = []  # for summary

    for use_search, r in user_choices:
        if use_search and r["item"]:
            # trust item
            ris_blocks.append(convert_item_to_ris(r["item"]))
            bib_blocks.append(convert_item_to_bibtex(r["item"]))
            final_results.append({"original": r["original"], "used": "search", "doi": r.get("doi")})
        else:
            # fallback to raw RIS
            ris_blocks.append(raw_ref_to_ris(r["original"]))
            bib_blocks.append("")  # optional: could craft minimal bib
            final_results.append({"original": r["original"], "used": "raw", "doi": None})

    # Show summary
    st.header("Summary")
    found_count = sum(1 for f in final_results if f["used"] == "search")
    raw_count = sum(1 for f in final_results if f["used"] == "raw")
    st.success(f"{found_count} references used search metadata, {raw_count} used raw->RIS fallback.")

    # Final export buttons
    if export_format == "RIS":
        final_ris = "".join(ris_blocks)
        st.download_button("Download RIS file", data=final_ris, file_name="references.ris", mime="application/x-research-info-systems")
        with st.expander("RIS Preview"):
            st.code(final_ris, language="text")
    else:
        final_bib = "".join(bib_blocks)
        if not final_bib.strip():
            st.warning("No BibTeX entries generated (many fallbacks were raw text).")
        st.download_button("Download BibTeX file", data=final_bib, file_name="references.bib", mime="text/x-bibtex")
        with st.expander("BibTeX Preview"):
            st.code(final_bib, language="text")

# Footer
st.caption("Search order: Crossref (title) → Semantic Scholar (fallback). Similarity uses simple string ratio to guard acceptance. Adjust threshold or manually pick for each reference.")
