# app.py
"""
Streamlit app: search Crossref + PubMed by title or DOI and convert results to RIS
Requirements: streamlit, requests, pypdf (optional for PDF), python-dateutil
Install with:
pip install streamlit requests python-dateutil
Run:
streamlit run app.py
"""

import streamlit as st
import requests
import re
import io
import time
from typing import Optional, Dict, Any, Tuple, List
from xml.etree import ElementTree as ET
from dateutil import parser as dateparser

st.set_page_config(page_title="Title/DOI → Search (Crossref / PubMed) → RIS", layout="wide")

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")
CROSSREF_API = "https://api.crossref.org/works"
EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

st.title("🔎 Title/DOI → Crossref & PubMed → RIS")
st.markdown(
    "Paste one **title or DOI per line**. App will try Crossref (DOI exact or title search) then PubMed, show found metadata and let you download results as **RIS**."
)

with st.expander("Example inputs (copy/paste)"):
    st.code("""10.1002/jbm.b.30864
Recent advances in silicone pressure-sensitive adhesives
Tissues and bone adhesives – historical aspects""")

inp = st.text_area("Titles or DOIs (one per line)", height=220)
use_crossref = st.checkbox("Use Crossref (recommended)", value=True)
use_pubmed = st.checkbox("Use PubMed (fallback)", value=True)
timeout_seconds = st.slider("HTTP timeout (seconds)", 5, 30, 10)

def is_doi(s: str) -> Optional[str]:
    if not s:
        return None
    s = s.strip()
    m = DOI_RE.search(s)
    return m.group(0) if m else None

def crossref_lookup_by_doi(doi: str, timeout: int=10) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        url = f"{CROSSREF_API}/{requests.utils.requote_uri(doi)}"
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json().get("message", {})
        return crossref_item_to_meta(data), None
    except Exception as e:
        return None, str(e)

def crossref_search_title(title: str, timeout: int=10) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        params = {"query.title": title, "rows": 1}
        r = requests.get(CROSSREF_API, params=params, timeout=timeout)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None, None
        return crossref_item_to_meta(items[0]), None
    except Exception as e:
        return None, str(e)

def crossref_item_to_meta(item: Dict[str,Any]) -> Dict[str,Any]:
    authors = []
    for a in item.get("author", []) or []:
        fam = a.get("family", "")
        giv = a.get("given", "")
        if fam or giv:
            authors.append({"family": fam, "given": giv})
    title = ""
    if item.get("title"):
        if isinstance(item["title"], list):
            title = item["title"][0]
        else:
            title = item["title"]
    year = ""
    try:
        issued = item.get("issued", {}).get("date-parts", [[]])
        if issued and issued[0]:
            year = str(issued[0][0])
    except Exception:
        year = ""
    meta = {
        "title": title,
        "authors": authors,
        "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
        "year": year,
        "volume": str(item.get("volume","") or ""),
        "issue": str(item.get("issue","") or ""),
        "pages": item.get("page","") or "",
        "doi": item.get("DOI","") or ""
    }
    return meta

def pubmed_search_by_title(title: str, timeout: int=10) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        params = {"db": "pubmed", "term": title + "[Title]", "retmax": 1, "retmode": "xml"}
        r = requests.get(EUTILS_SEARCH, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        id_elems = root.findall(".//Id")
        if not id_elems:
            return None, None
        pmid = id_elems[0].text
        return pubmed_fetch_by_pmid(pmid, timeout=timeout)
    except Exception as e:
        return None, str(e)

def pubmed_search_by_doi(doi: str, timeout: int=10) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        params = {"db": "pubmed", "term": doi + "[AID]", "retmax": 1, "retmode": "xml"}
        r = requests.get(EUTILS_SEARCH, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        id_elems = root.findall(".//Id")
        if not id_elems:
            return None, None
        pmid = id_elems[0].text
        return pubmed_fetch_by_pmid(pmid, timeout=timeout)
    except Exception as e:
        return None, str(e)

def pubmed_fetch_by_pmid(pmid: str, timeout: int=10) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = requests.get(EUTILS_FETCH, params=params, timeout=timeout)
        r.raise_for_status()
        root = ET.fromstring(r.content)
        # Title
        atitle = root.find(".//ArticleTitle")
        title = atitle.text if atitle is not None else ""
        # Authors
        authors = []
        for author in root.findall(".//Author"):
            ln = author.find("LastName")
            fn = author.find("ForeName")
            if ln is not None:
                authors.append({"family": ln.text or "", "given": fn.text or ""})
        # Journal
        j = root.find(".//Journal/Title")
        journal = j.text if j is not None else ""
        # Year
        year = ""
        # PubDate may be year or medline date
        pubdate = root.find(".//PubDate")
        if pubdate is not None:
            # try Year
            y = pubdate.find("Year")
            if y is not None and y.text:
                year = y.text
            else:
                # find MedlineDate or try parse
                md = pubdate.find("MedlineDate")
                if md is not None and md.text:
                    try:
                        year = str(dateparser.parse(md.text, fuzzy=True).year)
                    except Exception:
                        year = ""
        # Volume/Issue/Pages
        vol = (root.find(".//Volume").text if root.find(".//Volume") is not None else "") or ""
        issue = (root.find(".//Issue").text if root.find(".//Issue") is not None else "") or ""
        pages = (root.find(".//MedlinePgn").text if root.find(".//MedlinePgn") is not None else "") or ""
        # DOI from ArticleIdList
        doi = ""
        for aid in root.findall(".//ArticleId"):
            if aid.get("IdType") == "doi":
                doi = aid.text or ""
        meta = {
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "volume": vol,
            "issue": issue,
            "pages": pages,
            "doi": doi
        }
        return meta, None
    except Exception as e:
        return None, str(e)

def meta_to_ris(meta: Dict[str,Any]) -> str:
    """Convert a metadata dict to a RIS string for a journal article."""
    lines = ["TY  - JOUR"]
    title = meta.get("title","")
    if title:
        lines.append(f"TI  - {title}")
    for a in meta.get("authors", []):
        if isinstance(a, dict):
            fam = a.get("family","").strip()
            giv = a.get("given","").strip()
            if giv:
                au = f"{fam}, {giv}"
            else:
                au = fam
        else:
            au = str(a)
        if au:
            lines.append(f"AU  - {au}")
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

# Processing user inputs
if st.button("Search & Convert"):
    if not inp or not inp.strip():
        st.warning("Please paste at least one title or DOI (one per line).")
        st.stop()

    lines = [l.strip() for l in inp.splitlines() if l.strip()]
    results = []
    progress = st.progress(0)
    for i, line in enumerate(lines, start=1):
        found_meta = None
        err = None
        doi = is_doi(line)
        # 1) Try Crossref by DOI
        if doi and use_crossref:
            meta, err = crossref_lookup_by_doi(doi, timeout=timeout_seconds)
            if meta:
                found_meta = meta
        # 2) If not found and Crossref enabled, try Crossref title search
        if not found_meta and use_crossref:
            meta, err = crossref_search_title(line, timeout=timeout_seconds)
            if meta:
                found_meta = meta
        # 3) If not found and PubMed enabled, try PubMed DOI search (if DOI) then title
        if not found_meta and use_pubmed and doi:
            meta, err = pubmed_search_by_doi(doi, timeout=timeout_seconds)
            if meta:
                found_meta = meta
        if not found_meta and use_pubmed:
            meta, err = pubmed_search_by_title(line, timeout=timeout_seconds)
            if meta:
                found_meta = meta

        # If still not found, create placeholder metadata with original text as title
        if not found_meta:
            found_meta = {
                "title": line,
                "authors": [],
                "journal": "",
                "year": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": doi or ""
            }

        ris = meta_to_ris(found_meta)
        results.append({"input": line, "meta": found_meta, "ris": ris, "error": err})
        progress.progress(i / len(lines))
        time.sleep(0.05)

    # Show results
    st.success(f"Processed {len(results)} entries")
    combined_ris = ""
    for idx, r in enumerate(results, start=1):
        st.subheader(f"{idx}. Input: {r['input']}")
        if r["error"]:
            st.error(f"Warning (HTTP/parse error): {r['error']}")
        m = r["meta"]
        col1, col2 = st.columns([2,3])
        with col1:
            st.markdown("**Found metadata**")
            st.write("Title:", m.get("title",""))
            st.write("Journal:", m.get("journal",""))
            st.write("Year:", m.get("year",""))
            if m.get("doi"):
                st.write("DOI:", m.get("doi"))
        with col2:
            st.markdown("**Authors (preview)**")
            authors_preview = []
            for a in m.get("authors", [])[:12]:
                if isinstance(a, dict):
                    authors_preview.append(f"{a.get('family','')}, {a.get('given','')}".strip().strip(','))
                else:
                    authors_preview.append(str(a))
            st.write(authors_preview or "—")
        with st.expander("RIS for this entry"):
            st.code(r["ris"], language="text")
        combined_ris += r["ris"]

    # Download button for combined RIS
    st.markdown("---")
    st.download_button("Download combined RIS", data=combined_ris, file_name="references.ris", mime="application/x-research-info-systems")
    st.caption("Notes: Crossref often gives best structured metadata if DOI or exact title match. PubMed is used as alternate source for biomedical literature.")
