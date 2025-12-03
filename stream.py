import os
import re
import json
import time
import io
import requests
import streamlit as st
from typing import List, Dict, Any
from difflib import SequenceMatcher
from pypdf import PdfReader

# -------------------------
# Config
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")  # or set your key here directly
OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_THRESHOLD = 0.3

API_CACHE: Dict[str, Any] = {}

def cached(key: str, fn, *args, **kwargs):
    if key in API_CACHE:
        return API_CACHE[key]
    res = fn(*args, **kwargs)
    API_CACHE[key] = res
    return res

# -------------------------
# Utilities
# -------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

def extract_text_from_pdf(file_obj) -> str:
    try:
        reader = PdfReader(file_obj)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"ERROR_PDF_EXTRACT: {str(e)}"

# -------------------------
# OpenAI Parsing
# -------------------------
def openai_parse_reference(ref_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        st.error("OpenAI API key is required!")
        return {}
    
    prompt = (
        f"Extract bibliographic metadata from this reference:\n\n\"\"\"\n{ref_text}\n\"\"\"\n\n"
        "Return JSON only with: authors (list), title, journal, year, volume, issue, pages, doi."
    )
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    data = {"model": OPENAI_MODEL, "messages":[{"role":"user","content":prompt}], "temperature":0, "max_tokens":600}
    
    for attempt in range(3):  # retry 3 times for rate limits
        try:
            r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=20)
            r.raise_for_status()
            content = r.json()["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            return parsed
        except requests.exceptions.HTTPError as e:
            if r.status_code == 429 and attempt < 2:
                time.sleep(2 ** attempt)
                continue
            else:
                st.error(f"OpenAI parsing failed: {str(e)}")
                return {}
        except Exception as e:
            st.error(f"OpenAI parsing failed: {str(e)}")
            return {}

# -------------------------
# Search Crossref & PubMed
# -------------------------
def crossref_search(title_or_doi: str) -> Dict[str, Any]:
    def _fn(q):
        try:
            if title_or_doi.lower().startswith("10."):
                params = {"query.bibliographic": "", "filter": f"doi:{title_or_doi}", "rows":1}
            else:
                params = {"query.bibliographic": q, "rows":1}
            r = requests.get("https://api.crossref.org/works", params=params, timeout=10)
            r.raise_for_status()
            items = r.json().get("message", {}).get("items", [])
            if items:
                item = items[0]
                authors = [{"family": a.get("family",""), "given": a.get("given","")} for a in item.get("author",[])]
                return {
                    "title": item.get("title", [""])[0],
                    "authors": authors,
                    "journal": item.get("container-title", [""])[0],
                    "year": str(item.get("issued", {}).get("date-parts", [[""]])[0][0]),
                    "volume": str(item.get("volume","")),
                    "issue": str(item.get("issue","")),
                    "pages": item.get("page",""),
                    "doi": item.get("DOI","")
                }
            return {}
        except:
            return {}
    return cached(f"cr:{title_or_doi}", _fn, title_or_doi)

def pubmed_search(title: str) -> Dict[str, Any]:
    def _fn(q):
        try:
            params = {"db":"pubmed","term":q,"retmode":"json","retmax":1}
            r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=10)
            r.raise_for_status()
            ids = r.json().get("esearchresult", {}).get("idlist", [])
            if not ids: return {}
            r2 = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
                              params={"db":"pubmed","id":ids[0],"retmode":"xml"}, timeout=10)
            r2.raise_for_status()
            import xml.etree.ElementTree as ET
            root = ET.fromstring(r2.content)
            title_elem = root.find(".//ArticleTitle")
            title_text = title_elem.text if title_elem is not None else ""
            authors = []
            for a in root.findall(".//Author"):
                ln = a.find("LastName")
                fn = a.find("ForeName")
                if ln is not None:
                    authors.append({"family": ln.text or "", "given": fn.text or ""})
            journal_elem = root.find(".//Journal/Title")
            journal_text = journal_elem.text if journal_elem is not None else ""
            year_elem = root.find(".//PubDate/Year")
            year_text = year_elem.text if year_elem is not None else ""
            volume_elem = root.find(".//Volume")
            volume_text = volume_elem.text if volume_elem is not None else ""
            issue_elem = root.find(".//Issue")
            issue_text = issue_elem.text if issue_elem is not None else ""
            pages_elem = root.find(".//MedlinePgn")
            pages_text = pages_elem.text if pages_elem is not None else ""
            return {"title":title_text,"authors":authors,"journal":journal_text,"year":year_text,
                    "volume":volume_text,"issue":issue_text,"pages":pages_text,"doi":""}
        except:
            return {}
    return cached(f"pm:{title}", _fn, title)

# -------------------------
# RIS Export
# -------------------------
def convert_to_ris(meta: Dict[str,Any]) -> str:
    lines = ["TY  - JOUR"]
    if meta.get("title"): lines.append(f"TI  - {meta['title']}")
    for a in meta.get("authors", []):
        lines.append(f"AU  - {a.get('family','')}, {a.get('given','')}")
    if meta.get("journal"): lines.append(f"JO  - {meta['journal']}")
    if meta.get("volume"): lines.append(f"VL  - {meta['volume']}")
    if meta.get("issue"): lines.append(f"IS  - {meta['issue']}")
    if meta.get("pages"): lines.append(f"SP  - {meta['pages']}")
    if meta.get("year"): lines.append(f"PY  - {meta['year']}")
    if meta.get("doi"): lines.append(f"DO  - {meta['doi']}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n\n"

# -------------------------
# Streamlit App
# -------------------------
st.set_page_config(page_title="Reference → RIS", layout="wide")
st.title("📚 Reference to RIS Generator")

st.markdown("""
Paste references directly or upload PDF(s). AI will extract titles and DOIs, then search PubMed and Crossref.
""")

col1, col2 = st.columns([3,1])
with col1:
    input_mode = st.radio("Input type", ["Paste references", "Upload PDF(s)"], horizontal=True)
    raw_text = ""
    if input_mode=="Paste references":
        raw_text = st.text_area("Paste references here:", height=300)
    else:
        files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)
        if files:
            blocks = []
            for f in files:
                txt = extract_text_from_pdf(f)
                if not txt.startswith("ERROR"):
                    blocks.append(txt)
            raw_text = "\n\n".join(blocks)
            st.text_area("Extracted text:", raw_text, height=200)

with col2:
    threshold = st.slider("Match threshold", 0.0, 1.0, DEFAULT_THRESHOLD, 0.01)
    st.write("Using model:", OPENAI_MODEL)
    if OPENAI_API_KEY: st.success("OpenAI key set")
    else: st.error("Set OPENAI_API_KEY")

if st.button("Process References") and raw_text.strip():
    # Split references using common patterns like [1], 1., (1)
    refs = re.split(r"\n?\s*(?:\[\d+\]|\d+\.\s*|\(\d+\))\s*", raw_text)
    refs = [r.strip() for r in refs if r.strip()]
    
    processed = []
    progress = st.progress(0)
    for i, ref in enumerate(refs,1):
        ai_meta = openai_parse_reference(ref)
        title = ai_meta.get("title","")
        doi = ai_meta.get("doi","")
        found_meta = crossref_search(doi or title)
        if not found_meta:
            found_meta = pubmed_search(title)
        processed.append({"ref":ref, "ai":ai_meta, "found":found_meta})
        progress.progress(i/len(refs))
        time.sleep(0.1)
    
    st.session_state["processed_refs"] = processed
    st.success("Processing complete!")

if "processed_refs" in st.session_state:
    st.header("Review and Export")
    selections = []
    for idx, rec in enumerate(st.session_state["processed_refs"],1):
        st.subheader(f"Reference {idx}")
        st.markdown(f"**Original:** {rec['ref']}")
        st.markdown(f"**AI Extracted Title:** {rec['ai'].get('title','')}")
        st.markdown(f"**Found Title:** {rec['found'].get('title','')}")
        choice = st.radio(f"Choose metadata to export for ref {idx}", ["AI-extracted","Found"], index=1 if rec['found'] else 0, key=f"c{idx}")
        include = st.checkbox("Include in export", value=True, key=f"i{idx}")
        selections.append({"choice":choice,"include":include,"rec":rec})
    
    if st.button("Generate RIS Export"):
        ris_text = ""
        for s in selections:
            if not s["include"]: continue
            meta = s["rec"]["found"] if s["choice"]=="Found" and s["rec"]["found"] else s["rec"]["ai"]
            ris_text += convert_to_ris(meta)
        st.download_button("Download RIS", data=ris_text, file_name="references.ris", mime="application/x-research-info-systems")
        st.code(ris_text, language="text")
