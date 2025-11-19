# streamlit_ref_tool.py
"""
Reference → DOI → RIS/BibTeX Streamlit app
Search priority (two-pass PubMed):
  1. Crossref (title/biblio)
  2. PubMed (pass 1: full reference; pass 2: extracted title)
  3. Semantic Scholar (fallback)
If none match, parse the raw reference to extract fields and convert to RIS.
Save as streamlit_ref_tool.py and run:
  streamlit run streamlit_ref_tool.py

Requirements (requirements.txt):
  streamlit
  pypdf
  requests
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
# Utility helpers
# -------------------------
def normalize_text(s: str) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", s).strip()

def sim(a: str, b: str) -> float:
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
# Clean / split references
# -------------------------
def clean_and_join_broken_lines(text: str) -> str:
    if text is None:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # remove hyphenation
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # join single newlines inside paragraphs
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    # collapse spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()

def split_references_smart(text: str) -> List[str]:
    if not text:
        return []
    text = clean_and_join_broken_lines(text)

    # If numbered markers present, split on them
    if re.search(r"(?:^|\n)\s*(?:\[\d+\]|\d+[\.\)])\s+", text):
        parts = re.split(r"(?:\n\s*)?(?:\[\d+\]|\d+[\.\)])\s*", text)
        parts = [p.strip() for p in parts if len(p.strip()) > 10]
        if parts:
            return parts

    # Heuristic split on ". " followed by capital letter or bracket, conservative
    cand = re.split(r"\.\s+(?=[A-Z\[])", text)
    results = []
    for c in cand:
        c = c.strip()
        if not c:
            continue
        if not c.endswith("."):
            c = c + "."
        # very short fragments appended to previous
        if len(c) < 30 and results:
            results[-1] = results[-1].rstrip(".") + " " + c
        else:
            results.append(c)
    # merge fragments without year into previous
    final = []
    for r in results:
        if final and (len(r) < 60 and not re.search(r"\b(19|20)\d{2}\b", r)):
            final[-1] = final[-1].rstrip(".") + " " + r
        else:
            final.append(r)
    final = [re.sub(r"\s+", " ", f).strip() for f in final if f.strip()]
    return final

# -------------------------
# Raw reference parser (regex heuristics)
# Attempts to extract: authors, year, title, journal, volume, issue, pages, DOI
# -------------------------
def parse_raw_reference(ref: str) -> Dict[str, Any]:
    r = {"authors": [], "year": None, "title": "", "journal": "", "volume": "", "issue": "", "pages": "", "doi": None}
    text = ref.strip()

    # Try DOI
    doi_m = DOI_RE.search(text)
    if doi_m:
        r["doi"] = doi_m.group(0).rstrip(".,;")

    # Year (common patterns)
    year_m = re.search(r"\b(19|20)\d{2}\b", text)
    if year_m:
        r["year"] = year_m.group(0)

    # Authors: assume at start until a period before title
    # e.g. "Smith J, Jones A. Title..."
    parts = re.split(r"\.\s+", text, maxsplit=1)
    if len(parts) >= 2:
        maybe_authors = parts[0]
        rest = parts[1]
        # if authors contain commas and initials
        if re.search(r"[A-Z][a-z]+(?:\s+[A-Z]\.)?", maybe_authors):
            # split authors by semicolon or comma between name groups
            authors = re.split(r";\s*|,\s*(?=[A-Z][a-z])", maybe_authors)
            authors = [a.strip().rstrip('.') for a in authors if a.strip()]
            r["authors"] = authors
            # The rest likely contains title + journal; try to extract title
            # Title often ends before journal and year; look for journal abbreviations with vol/pages or year after
            # attempt to find journal by searching for pattern like "JournalName." or volume markers like "2005;17(3):811-21"
            # split rest by year or volume markers
            m = re.search(r"(.+?)\s+([A-Z][A-Za-z\s]{2,}\.?)\s+(\d{4}|[\d]{1,3}[:;])", rest)
            # fallback: split by journal separators like ";"
            title_guess = rest
            # If year exists in rest, cut off after title
            if r["year"]:
                title_split = re.split(r"\b" + re.escape(r["year"]) + r"\b", rest, maxsplit=1)
                if title_split:
                    title_guess = title_split[0].strip()
            else:
                # look for journal-like pattern "JournalName. 2005;..."
                m2 = re.search(r"([A-Z][A-Za-z\s&\-:]{2,})\.\s*\d{4}", rest)
                if m2:
                    title_guess = rest.split(m2.group(0))[0].strip()
            # Remove trailing PMID/DOI fragments
            title_guess = re.sub(r"http\S+$", "", title_guess).strip()
            r["title"] = title_guess.strip().rstrip(".")
        else:
            # no clear author segment; fallback heuristic: look for quoted title
            q = re.search(r'\"([^\"]+)\"', text)
            if q:
                r["title"] = q.group(1)
            else:
                # as last resort, take up to first period as title
                r["title"] = parts[0].strip()
    else:
        # single-line with no clear period, try to extract title in quotes
        q = re.search(r'\"([^\"]+)\"', text)
        if q:
            r["title"] = q.group(1)
        else:
            # attempt to find pattern Title. Journal Year;Volume:Pages
            m = re.search(r"^(.*?)\s+[A-Z][a-z]+\.", text)
            r["title"] = m.group(1).strip() if m else text[:200]

    # Journal, volume, pages extraction (common Vancouver patterns)
    vol_m = re.search(r"(\d{4})\s*;\s*(\d+)\s*\(?(\d+)?\)?\s*:\s*([\d\-–]+)", text)
    if vol_m:
        # pattern like 2005;17(3):811-21
        r["year"] = vol_m.group(1)
        r["volume"] = vol_m.group(2)
        r["issue"] = vol_m.group(3) or ""
        r["pages"] = vol_m.group(4)
        # Try to find journal name before the year
        jmatch = re.search(r"([A-Za-z\s\.&\-]+)\.\s*" + re.escape(vol_m.group(0)), text)
        if jmatch:
            r["journal"] = jmatch.group(1).strip().rstrip(".")
    else:
        # alternate page pattern like 'J Oral Pathol. 1984;13(6):661-70.'
        m2 = re.search(r"([A-Za-z\.\s&\-]+)\.\s*(\d{4})\s*;\s*(\d+)\(?(\d+)?\)?:\s*([\d\-–]+)", text)
        if m2:
            r["journal"] = m2.group(1).strip().rstrip(".")
            r["year"] = m2.group(2)
            r["volume"] = m2.group(3)
            r["issue"] = m2.group(4) or ""
            r["pages"] = m2.group(5)

    # Final cleanup: strip extra whitespace
    for k in r:
        if isinstance(r[k], str):
            r[k] = r[k].strip()
    return r

# -------------------------
# Crossref search (title-based / DOI direct)
# -------------------------
def search_crossref(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
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

    # build title guess (cut before year)
    try:
        split_on = re.split(r"\b(19|20)\d{2}\b", ref_text)
        title_guess = split_on[0] if split_on else ref_text
        title_guess = re.sub(r"\bvol\.?.*$", "", title_guess, flags=re.I)
        title_guess = title_guess.strip()
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
# PubMed search (two-pass): full ref then extracted title
# Uses esearch + efetch
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
        return pubmed_fetch_details(pmid)
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
        return pubmed_fetch_details(pmid)
    except Exception:
        return None, None

def pubmed_fetch_details(pmid: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
    try:
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        r2 = requests.get(fetch_url, params={"db":"pubmed", "id":pmid, "retmode":"xml"}, timeout=12)
        r2.raise_for_status()
        xml = r2.text

        # parse fields (lightweight)
        title_m = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml, re.S)
        title = title_m.group(1).strip() if title_m else ""

        journal_m = re.search(r"<Journal>.*?<Title>(.*?)</Title>", xml, re.S)
        journal = journal_m.group(1).strip() if journal_m else ""

        year_m = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.S)
        year = int(year_m.group(1)) if year_m else 0

        # authors - multiple <AuthorList><Author><LastName> / <ForeName>
        authors = []
        for m in re.finditer(r"<Author>(.*?)</Author>", xml, re.S):
            block = m.group(1)
            last = re.search(r"<LastName>(.*?)</LastName>", block)
            fore = re.search(r"<ForeName>(.*?)</ForeName>", block)
            if last:
                fam = last.group(1).strip()
                giv = fore.group(1).strip() if fore else ""
                authors.append({"family": fam, "given": giv})

        # pages
        pages_m = re.search(r"<MedlinePgn>(.*?)</MedlinePgn>", xml)
        pages = pages_m.group(1).strip() if pages_m else ""

        # volume/issue
        vol_m = re.search(r"<Volume>(.*?)</Volume>", xml)
        issue_m = re.search(r"<Issue>(.*?)</Issue>", xml)
        vol = vol_m.group(1).strip() if vol_m else ""
        issue = issue_m.group(1).strip() if issue_m else ""

        # DOI
        doi_m = re.search(r'<ArticleId IdType="doi">(.+?)</ArticleId>', xml)
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
def search_semantic_scholar(ref_text: str) -> Tuple[Optional[Dict[str,Any]], Optional[str]]:
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
# Converters
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
        lines.append(f"AU  - {a.get('family','')}, {a.get('given','')}")
    if item.get("container-title"):
        lines.append(f"JO  - {item['container-title'][0]}")
    if item.get("volume"):
        lines.append(f"VL  - {item['volume']}")
    if item.get("issue"):
        lines.append(f"IS  - {item['issue']}")
    if item.get("page"):
        p = item["page"]
        if "-" in p:
            sp, ep = p.split("-", 1)
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

def raw_ref_to_ris_parsed(raw: str) -> str:
    parsed = parse_raw_reference(raw)
    # Build RIS from parsed fields
    lines = []
    lines.append("TY  - GEN")
    if parsed.get("title"):
        lines.append(f"TI  - {parsed['title']}")
    # authors
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
st.set_page_config(page_title="Reference → RIS (Crossref + PubMed + SemanticScholar)", layout="wide")
st.title("📚 Reference → DOI → RIS (Crossref + PubMed + Semantic Scholar)")
st.write("Paste references or upload PDF. For each reference the app will search Crossref, then PubMed (two-pass) and finally Semantic Scholar. If no match, the raw reference will be parsed and converted to RIS.")

mode = st.radio("Input method", ["Paste references", "Upload PDF"], horizontal=True)

raw_text = ""
if mode == "Paste references":
    raw_text = st.text_area("Paste references here (supports numbered forms like [1], 1., 1) etc.)", height=300)
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
                    m = re.search(r"(References|REFERENCES|Bibliography|BIBLIOGRAPHY)([\s\S]{100,200000})", txt)
                    block = m.group(2) if m else txt
                    parts.append(block)
        raw_text = "\n\n".join(parts)
        if raw_text:
            st.text_area("Extracted text", raw_text, height=200)

auto_accept = st.checkbox("Auto-accept search result when similarity >= threshold", value=True)
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
    st.info("Searching Crossref → PubMed (full ref then title) → Semantic Scholar (fallback).")

    all_results = []
    seen_keys = set()
    ris_blocks = []
    bib_blocks = []

    progress = st.progress(0)
    status = st.empty()

    for i, ref in enumerate(refs, start=1):
        status.text(f"Searching {i}/{len(refs)}")
        # 1. Crossref
        item, doi = search_crossref(ref)
        source = None
        if doi:
            source = "Crossref"

        # 2. PubMed two-pass
        if not doi:
            item_pm, doi_pm = pubmed_search_full(ref)
            if doi_pm:
                item, doi = item_pm, doi_pm
                source = "PubMed (full)"
            else:
                # try extracted title
                parsed = parse_raw_reference(ref)
                title_guess = parsed.get("title") or parsed.get("title","")
                if title_guess:
                    item_pm2, doi_pm2 = pubmed_search_title_only(title_guess)
                    if doi_pm2:
                        item, doi = item_pm2, doi_pm2
                        source = "PubMed (title)"

        # 3. Semantic Scholar fallback
        if not doi:
            item_ss, doi_ss = search_semantic_scholar(ref)
            if doi_ss or item_ss:
                item, doi = item_ss, doi_ss
                source = "SemanticScholar"

        # similarity check between pasted ref and found title
        found_title = item.get("title",[None])[0] if item else None
        similarity_score = sim(ref, found_title) if found_title else 0.0

        # dedupe key
        key = canonicalize_for_dedupe(item, ref)
        duplicate = key in seen_keys
        if not duplicate:
            seen_keys.add(key)

        all_results.append({
            "original": ref,
            "item": item,
            "doi": doi,
            "source": source,
            "sim": similarity_score,
            "duplicate": duplicate
        })
        progress.progress(i/len(refs))
        time.sleep(0.15)

    status.empty()
    progress.empty()

    # Interactive review & decision
    st.header("Review results — choose per reference")
    chosen_outputs = []
    for idx, res in enumerate(all_results, start=1):
        with st.expander(f"Ref {idx}: {res['original'][:200]}{'...' if len(res['original'])>200 else ''}"):
            if res["duplicate"]:
                st.info("Duplicate detected — will be skipped in final output.")
            if res["item"]:
                st.markdown(f"**Found ({res['source']}) — similarity {res['sim']:.2f}**")
                st.write("Title:", res["item"].get("title", [""])[0] if res["item"].get("title") else "")
                if res["item"].get("container-title"):
                    st.write("Journal:", res["item"].get("container-title", [""])[0])
                if res.get("doi"):
                    st.write("DOI:", res["doi"])
            else:
                st.warning("No metadata found.")

            default_choice = 1  # 1 -> use raw, 0 -> use found
            if res["item"] and auto_accept and res["sim"] >= threshold:
                default_choice = 0

            choice = st.radio(
                f"Action for reference {idx}",
                ("Use found metadata (if any)", "Convert raw pasted reference → RIS"),
                index=default_choice,
                key=f"choice_{idx}"
            )
            if res["duplicate"]:
                st.write("Duplicate — will be ignored in final export.")
                chosen_outputs.append({"use": False, "type":"duplicate", "res": res})
            else:
                if choice == "Use found metadata (if any)" and res["item"]:
                    chosen_outputs.append({"use": True, "type":"search", "res": res})
                else:
                    chosen_outputs.append({"use": True, "type":"raw", "res": res})

    # Build final exports based on choices
    for entry in chosen_outputs:
        if not entry["use"]:
            continue
        if entry["type"] == "search":
            item = entry["res"]["item"]
            if export_format == "RIS":
                ris_blocks.append(convert_item_to_ris(item))
            else:
                bib_blocks.append(convert_item_to_bibtex(item))
        elif entry["type"] == "raw":
            raw = entry["res"]["original"]
            ris_blocks.append(raw_ref_to_ris_parsed(raw))
            bib_blocks.append("")  # could build minimal bib if needed

    # Summary and downloads
    total_found = sum(1 for e in chosen_outputs if e["use"] and e["type"]=="search")
    total_raw = sum(1 for e in chosen_outputs if e["use"] and e["type"]=="raw")
    st.success(f"Prepared {total_found} search-based entries and {total_raw} raw->RIS entries.")

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
            st.warning("No BibTeX content (many entries may be raw fallback).")
        st.download_button("Download BibTeX", data=final_bib, file_name="references.bib", mime="text/x-bibtex")
        with st.expander("BibTeX preview"):
            st.code(final_bib, language="text")

st.caption("Search order: Crossref → PubMed (full ref then title) → Semantic Scholar. If nothing found, parsed raw reference is converted to RIS with extracted fields where possible.")

