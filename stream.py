import streamlit as st
import re
import requests
import json

# ---------------------------------------
# Utility: Clean text
# ---------------------------------------
def clean_reference_text(text):
    # Fix broken words split across lines
    text = re.sub(r"(\w+)-\s*\n(\w+)", r"\1\2", text)

    # Merge wrapped lines
    text = re.sub(r"\n", " ", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------
# AI Parser Helper — Split Title vs Journal
# ---------------------------------------
def split_title_and_journal(ref_text):
    text = " ".join(ref_text.split())

    # Detect year
    year_match = re.search(r"(19|20)\d{2}", text)
    if year_match:
        year_pos = year_match.start()
        left = text[:year_pos].strip()
    else:
        left = text

    # Try splitting on period (title ends before journal)
    parts = [p.strip() for p in left.split(".") if p.strip()]

    if len(parts) == 1:
        return parts[0], ""

    title = parts[0]
    journal = " ".join(parts[1:])

    # Remove lingering volume numbers from journal section
    journal = re.sub(r"\b\d+.*$", "", journal).strip()

    return title, journal


# ---------------------------------------
# AI Hybrid Parser — Extract metadata
# ---------------------------------------
def ai_parse_reference(ref_text):
    ref = clean_reference_text(ref_text)

    # Authors
    authors = []
    author_match = re.match(r"([^\.]+)\.", ref)
    if author_match:
        raw_authors = author_match.group(1)
        parts = re.split(r",|;", raw_authors)
        for p in parts:
            p = p.strip()
            if len(p.split()) <= 1:
                continue
            authors.append(p)

    # Title + Journal
    title, journal = split_title_and_journal(ref)

    # Year
    year = ""
    year_match = re.search(r"(19|20)\d{2}", ref)
    if year_match:
        year = year_match.group(0)

    # Volume
    volume = ""
    vol_match = re.search(r"\b(\d{1,4})\s*[:\(]", ref)
    if vol_match:
        volume = vol_match.group(1)

    # Pages
    pages = ""
    page_match = re.search(r":\s*(\d+[-–]\d+|\d+)", ref)
    if page_match:
        pages = page_match.group(1)

    return {
        "title": title,
        "journal": journal,
        "authors": authors,
        "year": year,
        "volume": volume,
        "pages": pages,
        "doi": ""
    }


# ---------------------------------------
# Crossref Search
# ---------------------------------------
def crossref_lookup(query):
    try:
        r = requests.get("https://api.crossref.org/works", params={"query": query, "rows": 1}, timeout=8)
        if r.status_code != 200:
            return None

        item = r.json()["message"]["items"][0]

        authors = []
        for a in item.get("author", []):
            name = a.get("family", "") + " " + a.get("given", "")
            authors.append(name.strip())

        return {
            "title": item.get("title", [""])[0],
            "journal": item.get("container-title", [""])[0],
            "authors": authors,
            "year": item.get("issued", {}).get("date-parts", [[None]])[0][0],
            "volume": item.get("volume", ""),
            "pages": item.get("page", ""),
            "doi": item.get("DOI", "")
        }

    except:
        return None


# ---------------------------------------
# PubMed Search
# ---------------------------------------
def pubmed_lookup(query):
    try:
        # Step 1: Get PubMed ID
        e = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmode": "json"},
            timeout=8
        )
        pmids = e.json()["esearchresult"]["idlist"]
        if not pmids:
            return None

        pmid = pmids[0]

        # Step 2: Fetch metadata
        f = requests.get(
            "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",
            params={"db": "pubmed", "id": pmid, "retmode": "xml"},
            timeout=8
        )
        xml = f.text

        # Extract title
        title = re.search(r"<ArticleTitle>(.*?)</ArticleTitle>", xml)
        title = title.group(1) if title else ""

        # Journal
        journal = re.search(r"<Title>(.*?)</Title>", xml)
        journal = journal.group(1) if journal else ""

        # Year
        year = re.search(r"<PubDate>.*?<Year>(\d{4})</Year>", xml, re.DOTALL)
        year = year.group(1) if year else ""

        # Pages
        pages = re.search(r"<MedlinePgn>(.*?)</MedlinePgn>", xml)
        pages = pages.group(1) if pages else ""

        # Volume
        volume = re.search(r"<Volume>(.*?)</Volume>", xml)
        volume = volume.group(1) if volume else ""

        # Authors
        authors = []
        for match in re.findall(r"<Author>.*?<LastName>(.*?)</LastName>.*?<ForeName>(.*?)</ForeName>", xml, re.DOTALL):
            authors.append(f"{match[0]} {match[1]}")

        # DOI
        doi = re.search(r"<ELocationID EIdType=\"doi\">(.*?)</ELocationID>", xml)
        doi = doi.group(1) if doi else ""

        return {
            "title": title,
            "journal": journal,
            "authors": authors,
            "year": year,
            "volume": volume,
            "pages": pages,
            "doi": doi
        }

    except:
        return None


# ---------------------------------------
# Semantic Scholar Search
# ---------------------------------------
def semantic_lookup(query):
    try:
        r = requests.get(f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit=1&fields=title,venue,year,authors,externalIds,publicationVenue", timeout=8)

        if r.status_code != 200:
            return None

        item = r.json()["data"][0]

        authors = [a["name"] for a in item.get("authors", [])]

        return {
            "title": item.get("title", ""),
            "journal": item.get("venue", ""),
            "authors": authors,
            "year": item.get("year", ""),
            "volume": "",
            "pages": "",
            "doi": item.get("externalIds", {}).get("DOI", "")
        }

    except:
        return None


# ---------------------------------------
# Convert Dict → RIS
# ---------------------------------------
def convert_to_ris(meta):
    ris = "TY  - JOUR\n"

    for a in meta["authors"]:
        ris += f"AU  - {a}\n"

    ris += f"TI  - {meta['title']}\n"
    ris += f"JO  - {meta['journal']}\n"
    ris += f"PY  - {meta['year']}\n"
    ris += f"VL  - {meta['volume']}\n"
    ris += f"SP  - {meta['pages']}\n"
    ris += f"DO  - {meta['doi']}\n"
    ris += "ER  -\n\n"

    return ris


# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.title("📚 Advanced Reference-to-RIS Converter (AI + PubMed + Crossref)")

text_input = st.text_area("Paste your references below:", height=250)

if st.button("Convert to RIS"):
    refs = re.split(r"\n\d+[\.\)]|\[\d+\]\s*", text_input)
    refs = [r.strip() for r in refs if len(r.strip()) > 3]

    ris_output = ""

    for ref in refs:
        st.write(f"### Processing Reference:\n{ref}")

        # Cleanup text
        cleaned = clean_reference_text(ref)

        # --- SEARCH PHASE ---
        meta = (
            crossref_lookup(cleaned)
            or pubmed_lookup(cleaned)
            or semantic_lookup(cleaned)
        )

        # --- AI FALLBACK ---
        if meta is None:
            st.warning("No metadata found online → using AI parser.")
            meta = ai_parse_reference(cleaned)

        # --- RIS GENERATION ---
        ris_output += convert_to_ris(meta)

    st.subheader("Generated RIS")
    st.code(ris_output)

    st.download_button(
        "Download RIS File",
        ris_output,
        "references.ris",
        "text/plain"
    )
