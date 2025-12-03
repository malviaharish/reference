import os
import re
import json
import time
import io
import requests
import streamlit as st
from typing import List, Dict, Any
from pypdf import PdfReader

st.set_page_config(page_title="Reference → RIS", layout="wide")

# -------------------------
# Config / Secrets
# -------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4o-mini"

# -------------------------
# Utilities
# -------------------------
def extract_text_from_pdf(file_obj) -> str:
    """Extract text from PDF file"""
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        return text
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {str(e)}"

def extract_individual_references(raw_text: str) -> List[str]:
    """
    Split references even if numbered as: [1], 1., 1), (1)
    """
    pattern = r"(?:(?:\[\d+\])|(?:\(\d+\))|(?:\d+\.)|(?:\d+\)))\s*"
    refs = re.split(pattern, raw_text)
    cleaned = [r.strip() for r in refs if len(r.strip()) > 5]
    return cleaned

# -------------------------
# OpenAI parsing (title + DOI)
# -------------------------
def openai_parse_reference(ref_text: str) -> Dict[str, str]:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found. Set OPENAI_API_KEY in Streamlit secrets.")
        return {"title": "", "doi": ""}
    
    prompt = f"""
Extract the **title** and **DOI** (if present) from this bibliographic reference. 
Return JSON in the format: {{ "title": "...", "doi": "..." }}. 
Reference: """ + ref_text

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0.0, "max_tokens":300}

    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=20)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {"title": parsed.get("title","").strip(), "doi": parsed.get("doi","").strip()}
    except Exception as e:
        st.error(f"OpenAI parsing error: {str(e)}")
        return {"title": "", "doi": ""}

# -------------------------
# Search PubMed & Crossref
# -------------------------
def search_pubmed(title: str) -> Dict[str, str]:
    if not title:
        return {}
    try:
        params = {"db":"pubmed", "term": title, "retmode":"json", "retmax":1}
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {}
        pmid = ids[0]
        # Fetch metadata
        r2 = requests.get(f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/?format=pubmed", timeout=10)
        r2.raise_for_status()
        # Minimal metadata; real implementation can parse XML for full
        return {"source":"PubMed", "pmid": pmid, "title": title}
    except:
        return {}

def search_crossref(title: str, doi: str="") -> Dict[str, str]:
    try:
        if doi:
            r = requests.get(f"https://api.crossref.org/works/{doi}", timeout=10)
            r.raise_for_status()
            data = r.json().get("message", {})
        else:
            params = {"query.title": title, "rows":1}
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if not items:
                return {}
            data = items[0]
        authors = []
        for a in data.get("author", []):
            fam = a.get("family","")
            giv = a.get("given","")
            authors.append(f"{fam}, {giv}" if giv else fam)
        return {
            "source":"Crossref",
            "title": data.get("title", [""])[0] if data.get("title") else "",
            "doi": data.get("DOI",""),
            "journal": data.get("container-title", [""])[0] if data.get("container-title") else "",
            "year": str(data.get("issued", {}).get("date-parts", [[""]])[0][0]) if data.get("issued") else "",
            "volume": data.get("volume",""),
            "issue": data.get("issue",""),
            "pages": data.get("page",""),
            "authors": authors
        }
    except:
        return {}

# -------------------------
# RIS converter
# -------------------------
def convert_to_ris(meta: Dict[str,Any]) -> str:
    lines = ["TY  - JOUR"]
    if meta.get("title"):
        lines.append(f"TI  - {meta['title']}")
    for a in meta.get("authors", []):
        lines.append(f"AU  - {a}")
    if meta.get("journal"):
        lines.append(f"JO  - {meta['journal']}")
    if meta.get("volume"):
        lines.append(f"VL  - {meta['volume']}")
    if meta.get("issue"):
        lines.append(f"IS  - {meta['issue']}")
    if meta.get("pages"):
        lines.append(f"SP  - {meta['pages']}")
    if meta.get("year"):
        lines.append(f"PY  - {meta['year']}")
    if meta.get("doi"):
        lines.append(f"DO  - {meta['doi']}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 Reference → RIS Generator")

col1, col2 = st.columns([3,1])

with col2:
    st.write("Settings")
    st.info("OpenAI key is required for AI parsing.")
    st.checkbox("Auto-accept found metadata", value=True)
    export_format = st.selectbox("Export format", ["RIS"])

with col1:
    input_method = st.radio("Input method", ["Paste references", "Upload PDF(s)"])
    raw_text = ""
    if input_method == "Paste references":
        raw_text = st.text_area("Paste references here:", height=300)
    else:
        files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if files:
            texts = []
            for f in files:
                t = extract_text_from_pdf(f)
                texts.append(t)
            raw_text = "\n\n".join(texts)
            st.text_area("Extracted text from PDF(s)", raw_text, height=200)

if st.button("Process References"):
    if not raw_text.strip():
        st.warning("Paste references or upload PDFs first.")
        st.stop()

    ref_list = extract_individual_references(raw_text)
    st.info(f"Found {len(ref_list)} references. Processing...")

    results = []
    progress = st.progress(0)
    for i, ref in enumerate(ref_list, 1):
        ai_data = openai_parse_reference(ref)
        crossref_data = search_crossref(ai_data.get("title",""), ai_data.get("doi",""))
        pubmed_data = search_pubmed(ai_data.get("title",""))
        # Prefer Crossref > PubMed > AI
        chosen = crossref_data or pubmed_data or ai_data
        ris_entry = convert_to_ris(chosen)
        results.append(ris_entry)
        progress.progress(i/len(ref_list))
        time.sleep(0.05)

    final_ris = "\n".join(results)
    st.download_button("Download RIS", data=final_ris, file_name="references.ris", mime="application/x-research-info-systems")
    st.text_area("RIS Preview", final_ris, height=400)

