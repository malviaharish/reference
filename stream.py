# app.py
import streamlit as st
import requests
import re
import time
from xml.etree import ElementTree as ET
from dateutil import parser as dateparser
from openai import OpenAI

# ----------------------------- CONFIG -----------------------------------

st.set_page_config(page_title="Reference Parser + Search + RIS Generator", layout="wide")

OPENAI_MODEL = "gpt-4o"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

CROSSREF_API = "https://api.crossref.org/works"
EUTILS_SEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EUTILS_FETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

DOI_RE = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+")

# ----------------------------- UI ---------------------------------------

st.title("📘 Paste References → AI Parsing → Crossref + PubMed → RIS Export")

st.markdown("""
### Paste *any* reference format:
- Numbered references  
- Multi-line references  
- Journals, books, conference papers  
- With or without DOI  

The app will **automatically extract** metadata using GPT-4o and then search online databases.
""")

example = """
[7] S.B. Lin, L.D. Durfee, R.A. Ekeland, J. Mcvie, G.K. Schalau, Recent advances in silicone pressure-sensitive adhesives, 
J. Adhes. Sci. Technol. 21 (2007) 605–623.

[8] M. Donkerwolcke, F. Burny, D. Muster, Tissues and bone adhesives-historical aspects, Biomaterials 19 (1998) 1461–1466.
"""

with st.expander("Example input"):
    st.code(example)

raw_input_text = st.text_area("Paste your references here:", height=300)

use_crossref = st.checkbox("Use Crossref", value=True)
use_pubmed = st.checkbox("Use PubMed", value=True)

# ----------------------------- FUNCTIONS ---------------------------------------

def ai_extract_metadata(ref_text: str):
    """Use GPT-4o to extract title, DOI, year, journal, etc."""
    prompt = f"""
Extract clean metadata from the following reference. Return JSON with the following keys:
title, authors (array of 'Family, Given'), journal, year, volume, issue, pages, doi (if available).

Reference:
{ref_text}
"""
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        import json
        return json.loads(response.choices[0].message["content"])
    except Exception as e:
        return {"title": ref_text, "authors": [], "journal": "", "year": "", "volume": "", "issue": "", "pages": "", "doi": ""}

def is_doi(s: str):
    m = DOI_RE.search(s)
    return m.group(0) if m else None


# ---- Crossref Lookup ----

def crossref_lookup_by_doi(doi: str):
    try:
        url = f"{CROSSREF_API}/{doi}"
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()["message"]
        return {
            "title": data["title"][0] if data.get("title") else "",
            "authors": [{"family": a.get("family", ""), "given": a.get("given", "")} for a in data.get("author", [])],
            "journal": data.get("container-title", [""])[0],
            "year": str(data.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "volume": data.get("volume", ""),
            "issue": data.get("issue", ""),
            "pages": data.get("page", ""),
            "doi": data.get("DOI", "")
        }
    except:
        return None

def crossref_search_title(title: str):
    try:
        r = requests.get(CROSSREF_API, params={"query.title": title, "rows": 1})
        r.raise_for_status()
        items = r.json()["message"]["items"]
        if not items:
            return None
        item = items[0]
        return {
            "title": item["title"][0] if item.get("title") else "",
            "authors": [{"family": a.get("family", ""), "given": a.get("given", "")} for a in item.get("author", [])],
            "journal": item.get("container-title", [""])[0],
            "year": str(item.get("issued", {}).get("date-parts", [[None]])[0][0]),
            "volume": item.get("volume", ""),
            "issue": item.get("issue", ""),
            "pages": item.get("page", ""),
            "doi": item.get("DOI", "")
        }
    except:
        return None


# ---- PubMed ----

def pubmed_search_title(title: str):
    try:
        r = requests.get(EUTILS_SEARCH, params={"db": "pubmed", "term": title + "[Title]", "retmax": 1, "retmode": "xml"})
        root = ET.fromstring(r.text)
        idnode = root.find(".//Id")
        if idnode is None:
            return None
        return pubmed_fetch(idnode.text)
    except:
        return None

def pubmed_fetch(pmid: str):
    r = requests.get(EUTILS_FETCH, params={"db": "pubmed", "id": pmid, "retmode": "xml"})
    root = ET.fromstring(r.text)
    title = root.find(".//ArticleTitle")
    journal = root.find(".//Journal/Title")
    year = root.find(".//PubDate/Year")

    meta = {
        "title": title.text if title is not None else "",
        "journal": journal.text if journal is not None else "",
        "year": year.text if year is not None else "",
        "authors": [],
        "volume": "",
        "issue": "",
        "pages": "",
        "doi": ""
    }

    for a in root.findall(".//Author"):
        ln = a.find("LastName")
        fn = a.find("ForeName")
        if ln is not None:
            meta["authors"].append({"family": ln.text, "given": fn.text if fn is not None else ""})

    for aid in root.findall(".//ArticleId"):
        if aid.get("IdType") == "doi":
            meta["doi"] = aid.text

    return meta


# ---- RIS Converter ----
def to_ris(meta):
    out = ["TY  - JOUR"]
    if meta.get("title"): out.append(f"TI  - {meta['title']}")
    for a in meta.get("authors", []):
        out.append(f"AU  - {a['family']}, {a['given']}")
    if meta.get("journal"): out.append(f"JO  - {meta['journal']}")
    if meta.get("year"): out.append(f"PY  - {meta['year']}")
    if meta.get("volume"): out.append(f"VL  - {meta['volume']}")
    if meta.get("issue"): out.append(f"IS  - {meta['issue']}")
    if meta.get("pages"): out.append(f"SP  - {meta['pages']}")
    if meta.get("doi"): out.append(f"DO  - {meta['doi']}")
    out.append("ER  - ")
    return "\n".join(out) + "\n\n"

# ----------------------------- PROCESS ---------------------------------------

if st.button("Parse + Search + Convert"):
    if not raw_input_text.strip():
        st.error("Please paste some references.")
        st.stop()

    references = [r.strip() for r in raw_input_text.split("\n\n") if r.strip()]

    all_ris = ""
    progress = st.progress(0)

    for i, ref in enumerate(references, start=1):
        st.subheader(f"Reference {i}")

        # -------- Step 1: AI Metadata extraction --------
        ai_meta = ai_extract_metadata(ref)
        st.write("### 🔍 AI Extracted Metadata")
        st.json(ai_meta)

        title = ai_meta.get("title", "")
        doi = ai_meta.get("doi", "")
        final_meta = None

        # -------- Step 2: Crossref by DOI --------
        if doi and use_crossref:
            final_meta = crossref_lookup_by_doi(doi)

        # -------- Step 3: Crossref by Title --------
        if not final_meta and use_crossref and title:
            final_meta = crossref_search_title(title)

        # -------- Step 4: PubMed Search --------
        if not final_meta and use_pubmed:
            final_meta = pubmed_search_title(title)

        # -------- Step 5: If still nothing, use AI metadata --------
        if not final_meta:
            final_meta = ai_meta

        st.write("### 📘 Final Merged Metadata")
        st.json(final_meta)

        ris_entry = to_ris(final_meta)
        all_ris += ris_entry

        with st.expander(f"RIS for Reference {i}"):
            st.code(ris_entry, language="text")

        progress.progress(i / len(references))

    st.success("All references processed!")
    st.download_button("⬇️ Download All as RIS", data=all_ris, file_name="references.ris")

