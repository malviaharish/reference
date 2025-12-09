# app.py
"""
Streamlit: Paste full references → Search in Crossref & PubMed → Export RIS
Now includes comparison table: Pasted Reference vs Extracted Metadata
"""

import streamlit as st
import requests
import re
import time
from typing import Dict, Any, Optional, Tuple
from xml.etree import ElementTree as ET
from dateutil import parser as dateparser

# ----------------------------- CONFIG -------------------------------- #

st.set_page_config(page_title="Reference → Crossref/PubMed → RIS", layout="wide")

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

CROSSREF_API = "https://api.crossref.org/works"
EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ----------------------------- UI -------------------------------- #

st.title("📚 Full Reference Parser → Crossref & PubMed → RIS")
st.markdown("""
Paste **full references** (even with numbering like 1., (1), [1])  
The app will search metadata using the **entire reference string**.
""")

with st.expander("Example"):
    st.code("""
[7] S.B. Lin, L.D. Durfee... Recent advances in silicone pressure-sensitive adhesives...
[8] M. Donkerwolcke... Tissues and bone adhesives – historical aspects...
""")

inp = st.text_area("Paste references (one per line)", height=220)

use_crossref = st.checkbox("Use Crossref", True)
use_pubmed = st.checkbox("Use PubMed", True)

timeout_seconds = st.slider("HTTP timeout", 5, 30, 10)


# ------------------------ DOI DETECTION ----------------------------- #

def extract_doi(text: str) -> Optional[str]:
    if not text:
        return None
    m = DOI_RE.search(text)
    return m.group(0) if m else None


# ------------------------ CROSSREF SEARCH --------------------------- #

def crossref_lookup(doi: str):
    try:
        url = f"{CROSSREF_API}/{doi}"
        r = requests.get(url, timeout=timeout_seconds)
        r.raise_for_status()
        msg = r.json()["message"]
        meta = crossref_to_meta(msg)
        meta["pubmed_id"] = ""  # 🔵 MODIFIED
        return meta, None
    except Exception as e:
        return None, str(e)


def crossref_fulltext_search(ref: str):
    try:
        params = {"query.bibliographic": ref, "rows": 1}
        r = requests.get(CROSSREF_API, params=params, timeout=timeout_seconds)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None, None
        meta = crossref_to_meta(items[0])
        meta["pubmed_id"] = ""  # 🔵 MODIFIED
        return meta, None
    except Exception as e:
        return None, str(e)


def crossref_to_meta(item: dict) -> dict:
    authors = []
    for a in item.get("author", []) or []:
        authors.append({"family": a.get("family", ""), "given": a.get("given", "")})

    title = item.get("title", [""])[0] if isinstance(item.get("title"), list) else item.get("title", "")
    journal = item.get("container-title", [""])[0] if item.get("container-title") else ""

    year = ""
    try:
        dp = item.get("issued", {}).get("date-parts", [[]])
        year = str(dp[0][0])
    except:
        pass

    return {
        "title": title,
        "authors": authors,
        "journal": journal,
        "year": year,
        "volume": item.get("volume", ""),
        "issue": item.get("issue", ""),
        "pages": item.get("page", ""),
        "doi": item.get("DOI", ""),
    }


# ------------------------ PUBMED SEARCH ----------------------------- #

def pubmed_search(ref: str):
    try:
        params = {"db": "pubmed", "term": ref, "retmax": 1, "retmode": "xml"}
        r = requests.get(EUTILS_SEARCH, params=params, timeout=timeout_seconds)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        ids = root.findall(".//Id")
        if not ids:
            return None, None

        pmid = ids[0].text
        meta, err = pubmed_fetch(pmid)

        if meta:
            meta["pubmed_id"] = pmid  # 🔵 MODIFIED

        return meta, err

    except Exception as e:
        return None, str(e)


def pubmed_fetch(pmid: str):
    try:
        params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
        r = requests.get(EUTILS_FETCH, params=params, timeout=timeout_seconds)
        r.raise_for_status()

        root = ET.fromstring(r.content)
        title = root.findtext(".//ArticleTitle", "")

        authors = []
        for a in root.findall(".//Author"):
            authors.append({
                "family": a.findtext("LastName", ""),
                "given": a.findtext("ForeName", "")
            })

        journal = root.findtext(".//Journal/Title", "")
        
        year = ""
        pd = root.find(".//PubDate")
        if pd is not None:
            y = pd.find("Year")
            if y is not None:
                year = y.text
            else:
                md = pd.find("MedlineDate")
                if md is not None:
                    try:
                        year = str(dateparser.parse(md.text, fuzzy=True).year)
                    except:
                        pass

        doi = ""
        for aid in root.findall(".//ArticleId"):
            if aid.attrib.get("IdType") == "doi":
                doi = aid.text

        return {
            "title": title,
            "authors": authors,
            "journal": journal,
            "year": year,
            "volume": root.findtext(".//Volume", ""),
            "issue": root.findtext(".//Issue", ""),
            "pages": root.findtext(".//MedlinePgn", ""),
            "doi": doi,
        }, None

    except Exception as e:
        return None, str(e)


# ------------------------ RIS FORMATTER ----------------------------- #

def to_ris(meta: dict) -> str:
    ris = ["TY  - JOUR"]

    if meta.get("title"):
        ris.append(f"TI  - {meta['title']}")

    for a in meta.get("authors", []):
        fam = a.get("family", "")
        giv = a.get("given", "")
        ris.append(f"AU  - {fam}, {giv}".rstrip(", "))

    if meta.get("journal"):
        ris.append(f"JO  - {meta['journal']}")
    if meta.get("year"):
        ris.append(f"PY  - {meta['year']}")
    if meta.get("volume"):
        ris.append(f"VL  - {meta['volume']}")
    if meta.get("issue"):
        ris.append(f"IS  - {meta['issue']}")
    if meta.get("pages"):
        ris.append(f"SP  - {meta['pages']}")
    if meta.get("doi"):
        ris.append(f"DO  - {meta['doi']}")
    if meta.get("pubmed_id"):  # 🔵 MODIFIED
        ris.append(f"ID  - PMID:{meta['pubmed_id']}")

    ris.append("ER  - ")
    return "\n".join(ris) + "\n\n"


# ------------------------ PROCESS ----------------------------- #

if st.button("Search & Convert"):

    if not inp.strip():
        st.warning("Paste references!")
        st.stop()

    lines = [l.strip() for l in inp.split("\n") if l.strip()]
    combined_ris = ""
    progress = st.progress(0)

    for i, ref in enumerate(lines, 1):
        doi = extract_doi(ref)
        result = None
        err = None

        if doi and use_crossref:
            result, err = crossref_lookup(doi)

        if not result and use_crossref:
            result, err = crossref_fulltext_search(ref)

        if not result and use_pubmed:
            result, err = pubmed_search(ref)

        if not result:
            result = {
                "title": ref,
                "authors": [],
                "journal": "",
                "year": "",
                "volume": "",
                "issue": "",
                "pages": "",
                "doi": doi or "",
                "pubmed_id": ""      # 🔵 MODIFIED
            }

        # -------------------- EDITABLE FIELDS (NEW) -------------------- #
        st.subheader(f"Entry #{i}")

        with st.form(f"edit_{i}"):   # 🔵 MODIFIED
            colA, colB = st.columns(2)

            with colA:
                title = st.text_input("Title", result.get("title", ""))
                journal = st.text_input("Journal", result.get("journal", ""))
                year = st.text_input("Year", result.get("year", ""))
                doi_field = st.text_input("DOI", result.get("doi", ""))

            with colB:
                pmid = st.text_input("PubMed ID", result.get("pubmed_id", ""))
                volume = st.text_input("Volume", result.get("volume", ""))
                issue = st.text_input("Issue", result.get("issue", ""))
                pages = st.text_input("Pages", result.get("pages", ""))

            save_btn = st.form_submit_button("Update Metadata")

        if save_btn:
            result["title"] = title
            result["journal"] = journal
            result["year"] = year
            result["doi"] = doi_field
            result["pubmed_id"] = pmid
            result["volume"] = volume
            result["issue"] = issue
            result["pages"] = pages

        # Display table
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### **Pasted Reference**")
            st.code(ref, language="text")

        with col2:
            st.markdown("### **Extracted Metadata (Editable Above)**")
            st.write(f"**Title:** {result['title']}")
            st.write(f"**Journal:** {result['journal']}")
            st.write(f"**Year:** {result['year']}")
            st.write(f"**DOI:** {result['doi']}")
            st.write(f"**PubMed ID:** {result['pubmed_id']}")
            st.write(f"**Volume:** {result['volume']}")
            st.write(f"**Issue:** {result['issue']}")
            st.write(f"**Pages:** {result['pages']}")

        # RIS
        ris = to_ris(result)
        combined_ris += ris

        with st.expander("RIS"):
            st.code(ris, language="text")

        if err:
            st.warning(f"Lookup warning: {err}")

        progress.progress(i / len(lines))
        time.sleep(0.05)

    st.download_button("Download Combined RIS", combined_ris, "references.ris")
    st.success("Done!")
