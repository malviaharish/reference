import streamlit as st
import re
from typing import Dict

st.title("Reference Metadata Extractor with Editable Fields + PubMed ID")

# -----------------------------
# Function to extract metadata
# -----------------------------
def extract_metadata(ref: str) -> Dict[str, str]:
    metadata = {}

    # Extract title – text inside quotes ‘ ’ or " "
    title_match = re.search(r"[‘'“\"](.+?)[’'”\"]", ref)
    metadata["Title"] = title_match.group(1) if title_match else ""

    # Extract journal name – word-like phrases before a comma
    journal_match = re.search(r",\s*([^,]+?),", ref)
    metadata["Journal"] = journal_match.group(1).strip() if journal_match else ""

    # Extract year – first 4-digit year
    year_match = re.search(r"(19|20)\d{2}", ref)
    metadata["Year"] = year_match.group(0) if year_match else ""

    # Extract DOI
    doi_match = re.search(r"(10\.\S+)", ref)
    metadata["DOI"] = doi_match.group(1) if doi_match else ""

    # Extract PubMed ID (PMID)
    pmid_match = re.search(r"PMID[:\s]*([0-9]+)", ref, re.IGNORECASE)
    metadata["PubMed ID"] = pmid_match.group(1) if pmid_match else ""

    return metadata


# -----------------------------
# Input area
# -----------------------------
ref_input = st.text_area("Paste your reference here", height=150)

if ref_input:
    # Extract metadata
    extracted = extract_metadata(ref_input)

    st.markdown("## Extracted Metadata (Editable)")

    # Editable form for user modification
    with st.form("edit_form"):
        title = st.text_input("Title", extracted.get("Title", ""))
        journal = st.text_input("Journal", extracted.get("Journal", ""))
        year = st.text_input("Year", extracted.get("Year", ""))
        doi = st.text_input("DOI", extracted.get("DOI", ""))
        pmid = st.text_input("PubMed ID", extracted.get("PubMed ID", ""))

        submitted = st.form_submit_button("Update Metadata")

    # Updated metadata
    if submitted:
        extracted = {
            "Title": title,
            "Journal": journal,
            "Year": year,
            "DOI": doi,
            "PubMed ID": pmid
        }

    # -------------------------
    # Display final table
    # -------------------------
    st.markdown("## Output Table")

    table_md = f"""
| Pasted Reference | Extracted Metadata |
| ---------------- | ------------------ |
| {ref_input} | **Title:** {extracted["Title"]}<br>**Journal:** {extracted["Journal"]}<br>**Year:** {extracted["Year"]}<br>**DOI:** {extracted["DOI"]}<br>**PubMed ID:** {extracted["PubMed ID"]} |
"""
    st.markdown(table_md, unsafe_allow_html=True)
