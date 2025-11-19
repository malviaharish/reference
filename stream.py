# streamlit_ref_tool.py
import re
import time
import json
import hashlib
import urllib.parse
from typing import List, Tuple, Optional, Dict, Any

import requests
import streamlit as st
from pypdf import PdfReader

# ----------------------------
# Helpers: PDF extraction
# ----------------------------
def extract_text_from_pdf(uploaded_file) -> str:
    """Extract text from an uploaded PDF (file-like)."""
    try:
        reader = PdfReader(uploaded_file)
        parts = []
        for p in reader.pages:
            t = p.extract_text()
            if t:
                parts.append(t)
        return "\n".join(parts)
    except Exception as e:
        return f"ERROR: {e}"

# ----------------------------
# Helpers: cleaning & splitting
# ----------------------------
def clean_and_join_broken_lines(text: str) -> str:
    """
    Fix common PDF line-break artifacts:
    - Join hyphenated words at EOL
    - Merge lines with mid-sentence breaks
    """
    if not text:
        return ""
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove hyphenation at line breaks: "adhe\n-sives" variants
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Replace remaining single newlines between lowercase/uppercase with space
    text = re.sub(r"(?<=[^\n])\n(?=[^\n])", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    """
    Smart splitter for references:
    - Works for numbered and unnumbered lists
    - Joins wrapped lines
    - Uses heuristics to detect new reference starts
    """
    if not text:
        return []

    # First clean line breaks/hyphenation
    text = clean_and_join_broken_lines(text)

    # If clear numeric markers exist (1. or [1] or 1) ), split by them
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 30]
        if parts:
            return parts

    # Heuristic split for non-numbered references:
    # Split on period-space followed by Capitalized token that looks like an author surname or journal start
    # But be conservative: only split when the left segment contains typical citation cues (year, journal abbreviations, volume:page)
    cand = re.split(r"\.\s+(?=[A-Z])", text)
    results = []
    for c in cand:
        c = c.strip()
        if len(c) < 30:
            # likely stray; append to previous if exists
            if results:
                results[-1] = results[-1] + " " + c
            else:
                results.append(c)
        else:
            # ensure trailing period
            if not c.endswith("."):
                c = c + "."
            results.append(c)
    # Post-process: merge fragments that don't look like full references (no year/journal/pp)
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            # short fragment without year -> append to previous
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    # Final cleanup
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    # If still one long block with many commas, attempt splitting by double spaces / semicolons
    if len(final) == 1 and final[0].count(";") >= 3:
        alt = [s.strip() for s in re.split(r";\s*", final[0]) if len(s.strip()) > 30]
        if len(alt) > 1:
            final = alt
    return final

# ----------------------------
# Utilities: detection / validation
# ----------------------------
def detect_reference_style(text: str) -> Tuple[str, str]:
    """Simple heuristic to detect style and confidence."""
    text_lower = text.lower()
    score = {"Vancouver": 0, "APA": 0, "MLA": 0, "Chicago": 0}
    # Vancouver cues: year;volume:pages or year;vol
    if re.search(r"\b(19|20)\d{2}\b", text):
        score["Vancouver"] += len(re.findall(r"\b(19|20)\d{2}\b", text))
    if re.search(r"\(\d{4}\)", text):
        score["APA"] += len(re.findall(r"\(\d{4}\)", text))
    if '"' in text:
        score["MLA"] += text.count('"')
    best = max(score, key=score.get)
    conf = "Low"
    if score[best] >= 4:
        conf = "High"
    elif score[best] >= 1:
        conf = "Medium"
    # If all zero -> Unknown
    if all(v == 0 for v in score.values()):
        return "Unknown", "Low"
    return best, conf

def validate_reference(ref_text: str) -> Tuple[bool, List[str]]:
    """Very lightweight validation for user feedback."""
    issues = []
    if len(ref_text) < 20:
        issues.append("Reference seems very short")
    if not re.search(r"\b(19|20)\d{2}\b", ref_text):
        issues.append("No year found")
    if "," not in ref_text and "." not in ref_text:
        issues.append("No obvious separators found")
    return (len(issues) == 0), issues

# ----------------------------
# Crossref lookup
# ----------------------------
DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

def search_crossref(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    """
    Search Crossref: try DOI direct lookup; otherwise bibliographic query.
    Returns (item, doi) or (None, None)
    """
    # Try detect DOI in the text
    doi_match = DOI_RE.search(ref_text)
    try:
        if doi_match:
            doi = doi_match.group(0).rstrip(".,;")
            url = f"https://api.crossref.org/works/{doi}"
            r = requests.get(url, timeout=12)
            if r.status_code == 200:
                data = r.json().get("message", {})
                return data, doi
        # else use bibliographic search (use a short title guess)
        # extract a short chunk (before journal/year markers)
        title_guess = ref_text
        # attempt to cut after year/journal separators
        split_on = re.split(r"\b(19|20)\d{2}\b", ref_text)
        if split_on and len(split_on) > 1:
            title_guess = split_on[0]
        params = {"query.bibliographic": title_guess[:240], "rows": 1}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=12)
        r.raise_for_status()
        data = r.json().get("message", {})
        items = data.get("items", [])
        if items:
            it = items[0]
            return it, it.get("DOI")
    except Exception:
        return None, None
    return None, None

# ----------------------------
# PubMed lookup
# ----------------------------
def search_pubmed(ref_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Search PubMed via NCBI E-utilities (esearch + efetch).
    Returns a minimal item dict compatible with Crossref-converters.
    """
    try:
        base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db":"pubmed", "term": ref_text, "retmode":"json", "retmax":1}
        r = requests.get(base, params=params, timeout=12)
        r.raise_for_status()
        data = r.json()
        ids = data.get("esearchresult", {}).get("idlist", [])
        if not ids:
            return None, None
        pmid = ids[0]
        fetch = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r2 = requests.get(fetch, params={"db":"pubmed", "id": pmid, "retmode":"xml"}, timeout=12)
        r2.raise_for_status()
        xml = r2.text
        # parse simple fields
        doi_m = re.search(r'<ArticleId IdType="doi">([^<]+)</ArticleId>', xml)
        doi = doi_m.group(1) if doi_m else ""
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
        title = title_m.group(1).strip() if title_m else ""
        journal_m = re.search(r"<Title>(.*?)</Title>", xml)
        journal = journal_m.group(1).strip() if journal_m else ""
        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
        year = int(year_m.group(1)) if year_m else 0
        item = {
            "title":[title],
            "container-title":[journal],
            "DOI": doi,
            "issued": {"date-parts": [[year]]},
            "author": []
        }
        return item, doi
    except Exception:
        return None, None

# ----------------------------
# Google Scholar via SerpAPI (optional)
# ----------------------------
def search_google_scholar(ref_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    If you have SERPAPI_KEY in Streamlit secrets, this will use SerpAPI's Google Scholar engine.
    Otherwise it returns (None, None).
    """
    key = st.secrets.get("SERPAPI_KEY") if "SERPAPI_KEY" in st.secrets else None
    if not key:
        return None, None
    try:
        q = urllib.parse.quote_plus(ref_text)
        url = f"https://serpapi.com/search.json?q={q}&engine=google_scholar&api_key={key}"
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        org = data.get("organic_results", [])
        if not org:
            return None, None
        top = org[0]
        snippet = top.get("snippet", "")
        doi_m = DOI_RE.search(snippet)
        doi = doi_m.group(0) if doi_m else ""
        title = top.get("title", "")
        publication = top.get("publication", "")
        year = top.get("year") or 0
        item = {
            "title":[title],
            "container-title":[publication],
            "DOI": doi,
            "issued": {"date-parts": [[int(year) if year else 0]]},
            "author": []
        }
        return item, doi
    except Exception:
        return None, None

# ----------------------------
# Converters: RIS / BibTeX / EndNote / APA
# ----------------------------
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
        lines.append(f"AU  - {a.get('family','')}, {a.get('given','')}")
    if item.get("container-title"):
        lines.append(f"JO  - {item['container-title'][0]}")
    if item.get("issued", {}).get("date-parts"):
        lines.append(f"PY  - {item['issued']['date-parts'][0][0]}")
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
    if item.get("DOI"):
        lines.append(f"DO  - {item['DOI']}")
    if item.get("publisher"):
        lines.append(f"PB  - {item['publisher']}")
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
    btype = "article" if item.get("type","") == "journal-article" else "misc"
    authors = " and ".join([f"{a.get('family','')}, {a.get('given','')}" for a in (item.get("author") or [])])
    title = item.get("title", [""])[0]
    journal = item.get("container-title", [""])[0]
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
    """Fallback minimal RIS when no metadata found"""
    txt = raw.strip()
    if len(txt) > 250:
        txt = txt[:247] + "..."
    return f"TY  - GEN\nTI  - {txt}\nER  - \n\n"

# ----------------------------
# Deduplication
# ----------------------------
def canonicalize_for_dedupe(item: Optional[Dict[str,Any]], ref_text: str) -> str:
    if item and item.get("DOI"):
        return item["DOI"].lower()
    s = ref_text.lower()
    s = re.sub(r"\W+", " ", s)
    s = re.sub(r"\b(19|20)\d{2}\b", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Reference → RIS/BibTeX", layout="wide", page_icon="📚")

st.title("📚 Reference Finder & Exporter (Crossref → PubMed → Scholar → Raw→RIS)")
st.write("Upload a PDF or paste references. App will try Crossref → PubMed → Google Scholar (SerpAPI optional). If nothing found, raw text is converted to RIS.")

# Input
mode = st.radio("Input method:", ["Paste references", "Upload PDF"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references (multi-line). The parser supports numbered and unnumbered lists.", height=300)
else:
    uploaded = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
    if uploaded:
        all_pdf_text = []
        for f in uploaded:
            with st.spinner(f"Extracting {f.name} ..."):
                txt = extract_text_from_pdf(f)
                if txt.startswith("ERROR:"):
                    st.error(f"Error reading {f.name}: {txt}")
                else:
                    # try to find References section by keyword
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{50,200000})", txt)
                    if m:
                        block = m.group(2)
                    else:
                        block = txt
                    all_pdf_text.append(block)
        raw_text = "\n\n".join(all_pdf_text)
        if raw_text:
            st.text_area("Extracted text (from PDF)", raw_text, height=200)

export_format = st.selectbox("Export format:", ["RIS", "BibTeX"], index=0)
use_serpapi = "SERPAPI_KEY" in st.secrets

if use_serpapi:
    st.info("SerpAPI key found in Streamlit secrets. Google Scholar fallback enabled.")
else:
    st.info("No SerpAPI key found. Google Scholar fallback disabled (optional). Add 'SERPAPI_KEY' to secrets to enable).")

# Process
if st.button("Process & Generate Export"):
    if not raw_text.strip():
        st.warning("Please paste references or upload a PDF first.")
        st.stop()

    with st.spinner("Splitting references..."):
        refs = split_references_smart(raw_text)
    if not refs:
        st.warning("No references detected after splitting.")
        st.stop()
    st.success(f"Detected {len(refs)} references (after splitting).")

    # Detect style (show heuristic)
    style, conf = detect_reference_style(raw_text)
    st.info(f"Detected style: **{style}** (confidence: {conf})")

    # Process each with priority search & dedupe
    ris_blocks = []
    bib_blocks = []
    results = []  # list of dicts with original, found, doi, item, source
    seen_keys = set()

    progress = st.progress(0)
    status = st.empty()

    for i, ref in enumerate(refs, start=1):
        status.text(f"Searching {i}/{len(refs)}")
        # try Crossref
        item, doi = search_crossref(ref)
        source = None
        if doi:
            source = "Crossref"
        else:
            # try PubMed
            item, doi = search_pubmed(ref)
            if doi:
                source = "PubMed"
        if not doi and use_serpapi:
            item, doi = search_google_scholar(ref)
            if doi:
                source = "Scholar"

        # dedupe key
        key = canonicalize_for_dedupe(item, ref)
        if key in seen_keys:
            results.append({"original": ref, "found": bool(doi), "doi": doi or None, "item": item, "source": source, "status":"duplicate"})
            progress.progress(i/len(refs))
            time.sleep(0.2)
            continue
        seen_keys.add(key)

        if doi:
            # convert
            if export_format == "RIS":
                ris = convert_item_to_ris(item)
                ris_blocks.append(ris)
            else:
                bib = convert_item_to_bibtex(item)
                bib_blocks.append(bib)
            results.append({"original": ref, "found": True, "doi": doi, "item": item, "source": source, "status":"found"})
        else:
            # fallback raw -> ris
            ris = raw_ref_to_ris(ref)
            ris_blocks.append(ris)
            bib_blocks.append("")  # empty fallback for bib
            results.append({"original": ref, "found": False, "doi": None, "item": None, "source":"raw", "status":"fallback"})

        progress.progress(i/len(refs))
        time.sleep(0.25)

    status.empty()
    progress.empty()

    # Show duplicates summary
    duplicates = {}
    for idx, r in enumerate(results):
        if r["found"] and r["doi"]:
            duplicates.setdefault(r["doi"], []).append(idx+1)
    if duplicates:
        with st.expander("Duplicate DOIs detected"):
            for doi, idxs in duplicates.items():
                if len(idxs) > 1:
                    st.warning(f"DOI {doi} appears in references: {', '.join(map(str, idxs))}")

    # Show result list
    st.subheader("Processing summary")
    for idx, r in enumerate(results, start=1):
        if r["status"] == "duplicate":
            st.info(f"{idx}. [DUPLICATE] {r['original'][:150]}... (skipped)")
        elif r["status"] == "found":
            st.success(f"{idx}. [FOUND:{r['source']}] DOI: {r['doi']} — {r['original'][:150]}...")
        else:
            st.warning(f"{idx}. [FALLBACK RAW→RIS] {r['original'][:150]}...")

    # Aggregate output
    if export_format == "RIS":
        final_output = "".join(ris_blocks)
        if not final_output.strip():
            st.error("No RIS output generated.")
        else:
            st.download_button("Download RIS", data=final_output, file_name="references.ris", mime="application/x-research-info-systems")
            with st.expander("RIS Preview"):
                st.code(final_output, language="text")
    else:
        final_output = "".join(bib_blocks)
        if not final_output.strip():
            st.error("No BibTeX output generated.")
        else:
            st.download_button("Download BibTeX", data=final_output, file_name="references.bib", mime="text/x-bibtex")
            with st.expander("BibTeX Preview"):
                st.code(final_output, language="text")

    # store results for editing/regeneration (simple)
    st.session_state["last_results"] = results
    st.session_state["last_ris"] = final_output if export_format == "RIS" else None
    st.session_state["last_bib"] = final_output if export_format != "RIS" else None

# Optional: allow user to view/edit last results and regenerate export
if "last_results" in st.session_state:
    st.markdown("---")
    st.header("Review / Edit results")
    editable = []
    for idx, entry in enumerate(st.session_state["last_results"], start=1):
        with st.expander(f"Reference {idx} — {'FOUND' if entry['found'] else 'NOT FOUND'}"):
            st.write("Original:", entry["original"])
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input(f"title_{idx}", value=(entry["item"].get("title",[entry["original"]]) if entry["item"] else [entry["original"]])[0])
                journal = st.text_input(f"journal_{idx}", value=(entry["item"].get("container-title",[""])[0] if entry["item"] else ""))
                doi = st.text_input(f"doi_{idx}", value=(entry["doi"] or ""))
            with col2:
                year = st.text_input(f"year_{idx}", value=(str(entry["item"].get("issued",{}).get("date-parts",[[0]])[0][0]) if entry["item"] else ""))
                save = st.button(f"Save edits {idx}", key=f"save_{idx}")
                if save:
                    # write edits back into session state entry
                    new_item = entry["item"].copy() if entry["item"] else {"title":[title], "container-title":[journal], "issued": {"date-parts":[[int(year) if year.isdigit() else 0]]}}
                    new_item["title"] = [title]
                    new_item["container-title"] = [journal]
                    new_item["DOI"] = doi
                    try:
                        new_item["issued"] = {"date-parts":[[int(year)]]}
                    except Exception:
                        new_item["issued"] = {"date-parts":[[0]]}
                    st.session_state["last_results"][idx-1]["item"] = new_item
                    st.session_state["last_results"][idx-1]["doi"] = doi or None
                    st.success("Saved edits for entry " + str(idx))

    if st.button("Regenerate Export from Edited Results"):
        # regenerate based on session results
        ris_blocks = []
        bib_blocks = []
        for entry in st.session_state["last_results"]:
            it = entry.get("item")
            if export_format == "RIS":
                if it:
                    ris_blocks.append(convert_item_to_ris(it))
                else:
                    ris_blocks.append(raw_ref_to_ris(entry["original"]))
            else:
                if it:
                    bib_blocks.append(convert_item_to_bibtex(it))
                else:
                    bib_blocks.append("") 
        final = "".join(ris_blocks) if export_format == "RIS" else "".join(bib_blocks)
        st.session_state["last_ris" if export_format=="RIS" else "last_bib"] = final
        st.success("Regenerated export. Use the download button above to get the file.")
        if final:
            with st.expander("Preview regenerated output"):
                st.code(final, language="text")

st.caption("This app queries Crossref and PubMed. Google Scholar fallback via SerpAPI is optional. Add 'SERPAPI_KEY' in Streamlit secrets to enable Scholar search. For large batches consider adding a mailto param and/or longer delays.")
