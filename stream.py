import streamlit as st
import re
import requests
import json
import difflib
from PyPDF2 import PdfReader

# ------------------------------------------------------------
# Utility: Normalize text
# ------------------------------------------------------------

def normalize(text):
    return re.sub(r'\s+', ' ', text).strip().lower()


# ------------------------------------------------------------
# Extract references from pasted text
# Handles: numbered, [1], 1), broken lines, mixed formats
# ------------------------------------------------------------

def extract_references_from_text(input_text):
    lines = input_text.split("\n")
    refs = []
    current = ""

    for line in lines:
        if re.match(r"^\s*(\[\d+\]|\d+\.|\d+\)|\d+\s)", line):
            if current.strip():
                refs.append(current.strip())
            current = line.strip()
        else:
            current += " " + line.strip()

    if current.strip():
        refs.append(current.strip())

    return refs


# ------------------------------------------------------------
# Extract references from PDF
# ------------------------------------------------------------

def extract_references_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""

    for page in reader.pages:
        text += "\n" + page.extract_text()

    return extract_references_from_text(text)


# ------------------------------------------------------------
# Title similarity check
# ------------------------------------------------------------

def title_match(pasted, found):
    pasted_norm = normalize(pasted[:200])
    found_norm = normalize(found[:200])
    ratio = difflib.SequenceMatcher(None, pasted_norm, found_norm).ratio()
    return ratio > 0.35  # acceptance threshold


# ------------------------------------------------------------
# Crossref search (title search)
# ------------------------------------------------------------

def search_crossref_title(ref):
    try:
        title_guess = re.split(r"\.\s", ref, 1)[0]  # first sentence is usually title
        params = {"query.title": title_guess, "rows": 1}
        r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
        r.raise_for_status()
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return None, ""

        item = items[0]
        doi = item.get("DOI", "")

        # Final confirmation
        found_title = item.get("title", [""])[0]
        if not title_match(ref, found_title):
            return None, ""

        return item, doi
    except:
        return None, ""


# ------------------------------------------------------------
# PubMed search
# ------------------------------------------------------------

def search_pubmed(ref):
    try:
        # Step 1 → search
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {"db": "pubmed", "term": ref, "retmode": "json", "retmax": 1}
        r = requests.get(url, params=params, timeout=10)
        ids = r.json()["esearchresult"].get("idlist", [])
        if not ids:
            return None, ""

        pmid = ids[0]

        # Step 2 → fetch details
        url2 = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r2 = requests.get(url2, params={"db": "pubmed", "id": pmid, "retmode": "xml"}, timeout=10)
        xml = r2.text

        doi_match = re.search(r"<ArticleId IdType=\"doi\">(.+?)</ArticleId>", xml)
        doi = doi_match.group(1) if doi_match else ""

        title_match_xml = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml)
        title = title_match_xml.group(1) if title_match_xml else ""

        if not title_match(ref, title):
            return None, ""

        item = {
            "title": [title],
            "DOI": doi,
            "container-title": [""],
            "issued": {"date-parts": [[0]]},
            "author": []
        }

        return item, doi
    except:
        return None, ""


# ------------------------------------------------------------
# Google Scholar (SERPAPI optional)
# ------------------------------------------------------------

def search_scholar(ref):
    if "SERPAPI_KEY" not in st.secrets:
        return None, ""

    try:
        q = ref.replace(" ", "+")
        url = f"https://serpapi.com/search.json?engine=google_scholar&q={q}&api_key={st.secrets['SERPAPI_KEY']}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if "organic_results" not in data:
            return None, ""

        entry = data["organic_results"][0]
        title = entry.get("title", "")
        snippet = entry.get("snippet", "")

        doi_match = re.search(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", snippet)
        doi = doi_match.group(0) if doi_match else ""

        if not title_match(ref, title):
            return None, ""

        item = {
            "title": [title],
            "DOI": doi,
            "container-title": [""],
            "issued": {"date-parts": [[0]]},
            "author": []
        }
        return item, doi
    except:
        return None, ""


# ------------------------------------------------------------
# Fallback: Convert raw reference to minimal RIS
# ------------------------------------------------------------

def raw_to_ris(ref):
    return f"TY  - GEN\nTI  - {ref}\nER  -\n\n"


# ------------------------------------------------------------
# Convert Crossref/PubMed/Scholar record → RIS
# ------------------------------------------------------------

def to_ris(item):
    authors = item.get("author", [])
    ris = "TY  - JOUR\n"

    for a in authors:
        lname = a.get("family", "")
        fname = a.get("given", "")
        ris += f"AU  - {lname}, {fname}\n"

    ris += f"TI  - {item.get('title',[''])[0]}\n"
    ris += f"JO  - {item.get('container-title',[''])[0]}\n"
    ris += f"DO  - {item.get('DOI','')}\n"

    year = item.get("issued",{}).get("date-parts",[[0]])[0][0]
    ris += f"PY  - {year}\n"

    ris += "ER  -\n\n"
    return ris


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------

st.title("📚 Reference → DOI → RIS Generator")
st.write("Paste references or upload PDF. Supports any format, numbered or unnumbered.")

mode = st.radio("Choose input method", ["Paste Text", "Upload PDF"])

if mode == "Paste Text":
    text_input = st.text_area("Paste references here:")
    if st.button("Process"):
        refs = extract_references_from_text(text_input)

elif mode == "Upload PDF":
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])
    if pdf_file and st.button("Process"):
        refs = extract_references_from_pdf(pdf_file)
    else:
        refs = []


# ------------------------------------------------------------
# Processing
# ------------------------------------------------------------

if st.button("Generate RIS"):
    if not refs:
        st.error("No references found.")
        st.stop()

    all_ris = ""
    progress = st.progress(0)

    for i, ref in enumerate(refs):
        progress.progress((i + 1) / len(refs))

        # Try Crossref title search
        item, doi = search_crossref_title(ref)

        # Try PubMed
        if not doi:
            item, doi = search_pubmed(ref)

        # Try Scholar
        if not doi:
            item, doi = search_scholar(ref)

        # If nothing found → raw RIS
        if not doi:
            all_ris += raw_to_ris(ref)
        else:
            all_ris += to_ris(item)

    st.success("Done!")

    st.download_button(
        "Download RIS",
        data=all_ris,
        file_name="references.ris",
        mime="application/x-research-info-systems"
    )
