import os
import re
import json
import time
import requests
import streamlit as st
from typing import List, Dict, Any
from difflib import SequenceMatcher

st.set_page_config(page_title="Reference → RIS", layout="wide")

# -------------------------
# Config / secrets
# -------------------------
OPENAI_API_KEY = None
if "OPENAI_API_KEY" in st.secrets:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
else:
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

OPENAI_MODEL = "gpt-4o"  # change to "gpt-4" if available
DEFAULT_THRESHOLD = 0.3

# -------------------------
# Utilities
# -------------------------
def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, normalize_text(a).lower(), normalize_text(b).lower()).ratio()

def extract_refs(text: str) -> List[str]:
    """Split pasted text into individual references"""
    # Reference number patterns: 1, 1., [1], (1)
    pattern = r"(?:\n|^)\s*(?:\[\d+\]|\(\d+\)|\d+\.)\s*(.+?)(?=\n(?:\[\d+\]|\(\d+\)|\d+\.|\Z))"
    matches = re.findall(pattern, "\n" + text + "\n", flags=re.DOTALL)
    if matches:
        return [m.strip() for m in matches]
    else:
        return [text.strip()]

# -------------------------
# OpenAI parsing (optional)
# -------------------------
def openai_parse_reference(ref_text: str) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {"title": "", "doi": ""}
    
    prompt = (
        f"Extract the **title** and **DOI** from this bibliographic reference. "
        f"Return ONLY JSON like {{\"title\": \"...\", \"doi\": \"...\"}}.\n\nReference:\n{ref_text}"
    )
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {
            "model": OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=data, timeout=20)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        return {"title": parsed.get("title","").strip(), "doi": parsed.get("doi","").strip()}
    except Exception as e:
        st.warning(f"OpenAI parsing failed: {e}")
        return {"title": "", "doi": ""}

# -------------------------
# Crossref search
# -------------------------
def crossref_search(query: str, doi: str = "") -> Dict[str, Any]:
    if doi:
        url = f"https://api.crossref.org/works/{doi}"
        params = {}
    else:
        url = "https://api.crossref.org/works"
        params = {"query.bibliographic": query, "rows": 1}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        item = data.get("message") if doi else (data.get("message", {}).get("items", [None])[0])
        if not item:
            return {}
        authors = []
        for a in item.get("author", []):
            authors.append(f"{a.get('family','')}, {a.get('given','')}".strip())
        return {
            "title": item.get("title", [""])[0] if item.get("title") else "",
            "authors": authors,
            "journal": item.get("container-title", [""])[0] if item.get("container-title") else "",
            "year": str(item.get("issued", {}).get("date-parts", [[None]])[0][0]) if item.get("issued") else "",
            "volume": str(item.get("volume","")),
            "issue": str(item.get("issue","")),
            "pages": item.get("page",""),
            "doi": item.get("DOI","")
        }
    except Exception as e:
        st.warning(f"Crossref search failed: {e}")
        return {}

# -------------------------
# PubMed search by title
# -------------------------
def pubmed_search(title: str) -> Dict[str, Any]:
    if not title:
        return {}
    try:
        # Search for ID
        params = {"db": "pubmed", "term": title, "retmax": 1, "retmode": "json"}
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=params, timeout=10)
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return {}
        pmid = ids[0]
        # Fetch summary
        r2 = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                          params={"db":"pubmed","id":pmid,"retmode":"json"}, timeout=10)
        r2.raise_for_status()
        doc = r2.json().get("result", {}).get(pmid, {})
        authors = [f"{a.get('name','')}" for a in doc.get("authors", [])]
        return {
            "title": doc.get("title",""),
            "authors": authors,
            "journal": doc.get("source",""),
            "year": doc.get("pubdate","")[:4],
            "volume": doc.get("volume",""),
            "issue": doc.get("issue",""),
            "pages": doc.get("pages",""),
            "doi": doc.get("elocationid","") if "doi" in doc.get("elocationid","") else ""
        }
    except Exception as e:
        st.warning(f"PubMed search failed: {e}")
        return {}

# -------------------------
# Convert to RIS
# -------------------------
def convert_to_ris(meta: Dict[str, Any]) -> str:
    lines = ["TY  - JOUR"]
    if meta.get("title"): lines.append(f"TI  - {meta['title']}")
    for a in meta.get("authors", []):
        lines.append(f"AU  - {a}")
    if meta.get("journal"): lines.append(f"JO  - {meta['journal']}")
    if meta.get("volume"): lines.append(f"VL  - {meta['volume']}")
    if meta.get("issue"): lines.append(f"IS  - {meta['issue']}")
    if meta.get("pages"): lines.append(f"SP  - {meta['pages']}")
    if meta.get("year"): lines.append(f"PY  - {meta['year']}")
    if meta.get("doi"): lines.append(f"DO  - {meta['doi']}")
    lines.append("ER  - ")
    return "\n".join(lines) + "\n"

# -------------------------
# Process each reference
# -------------------------
def process_reference(ref: str) -> Dict[str, Any]:
    ai_data = openai_parse_reference(ref) if OPENAI_API_KEY else {"title":"", "doi":""}
    crossref_data = crossref_search(ai_data.get("title") or ref, ai_data.get("doi"))
    pubmed_data = pubmed_search(ai_data.get("title") or ref)
    # Pick the most complete result
    chosen = crossref_data if crossref_data else pubmed_data
    return {"original": ref, "ai": ai_data, "found": chosen}

# -------------------------
# Streamlit UI
# -------------------------
st.title("📚 Reference → RIS Exporter")

if not OPENAI_API_KEY:
    st.warning("⚠️ OpenAI key not found. AI parsing is disabled. The app will search Crossref and PubMed only.")

raw_text = st.text_area("Paste references here (numbered, e.g., 1, 1., [1])", height=300)

if st.button("Process References"):
    if not raw_text.strip():
        st.warning("Please paste references first.")
        st.stop()
    
    refs = extract_refs(raw_text)
    st.info(f"Found {len(refs)} reference(s). Processing...")

    results = []
    progress = st.progress(0)
    for i, r in enumerate(refs, start=1):
        rec = process_reference(r)
        results.append(rec)
        progress.progress(i/len(refs))
        time.sleep(0.1)
    
    st.success("Processing complete.")

    ris_entries = []
    for idx, rec in enumerate(results, start=1):
        st.markdown(f"### Reference {idx}")
        st.write("**Original:**", rec["original"])
        st.write("**AI-extracted:**", rec["ai"])
        st.write("**Found metadata:**", rec["found"])
        ris_entries.append(convert_to_ris(rec["found"] or {}))
    
    ris_text = "\n".join(ris_entries)
    st.download_button("Download RIS", data=ris_text, file_name="references.ris", mime="application/x-research-info-systems")
    st.text_area("RIS Preview", ris_text, height=400)
