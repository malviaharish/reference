import re
import requests
import streamlit as st
from pypdf import PdfReader   # <-- FIXED for Streamlit Cloud


# -----------------------------------------------------------
#               PDF TEXT EXTRACTION (STREAMLIT SAFE)
# -----------------------------------------------------------
def extract_text_from_pdf(uploaded_pdf):
    try:
        reader = PdfReader(uploaded_pdf)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"ERROR_READING_PDF: {e}"


# -----------------------------------------------------------
#      AUTO SPLITTER FOR NUMBERED / NON-NUMBERED REFERENCES
# -----------------------------------------------------------
def split_references_auto(text):
    """
    Splits references even if NOT numbered. Works for Vancouver style.

    Logic:
    - split on period followed by newline
    - if many lines belong together, merge them
    """

    # Normalize text
    text = re.sub(r"\s+", " ", text).strip()

    # Split on ". " followed by capital letter (new reference)
    # Example: "...811-21. Hess JR..."
    pattern = r"\.\s+(?=[A-Z])"

    refs = re.split(pattern, text)

    final_refs = []

    for r in refs:
        r = r.strip()
        if len(r) >= 25:       # avoid junk lines
            if not r.endswith("."):
                r += "."
            final_refs.append(r)

    return final_refs


# -----------------------------------------------------------
#               CROSSREF SEARCH
# -----------------------------------------------------------
def crossref_lookup(ref_text):
    url = "https://api.crossref.org/works"
    params = {"query.bibliographic": ref_text, "rows": 1}

    try:
        r = requests.get(url, params=params, timeout=10)
        data = r.json()

        items = data.get("message", {}).get("items", [])

        if items:
            return items[0], items[0].get("DOI", None)

        return None, None

    except:
        return None, None


# -----------------------------------------------------------
#                  RIS CONVERTER
# -----------------------------------------------------------
def item_to_ris(item):
    if item is None:
        return ""

    authors = item.get("author", [])
    ris = "TY  - JOUR\n"

    # Authors
    for a in authors:
        given = a.get("given", "")
        family = a.get("family", "")
        ris += f"AU  - {family}, {given}\n"

    title = item.get("title", [""])[0]
    jname = item.get("container-title", [""])[0]
    doi = item.get("DOI", "")
    year = ""

    try:
        year = str(item["issued"]["date-parts"][0][0])
    except:
        year = ""

    ris += (
        f"TI  - {title}\n"
        f"JO  - {jname}\n"
        f"PY  - {year}\n"
        f"DO  - {doi}\n"
        "ER  - \n\n"
    )

    return ris


# -----------------------------------------------------------
#                     STREAMLIT UI
# -----------------------------------------------------------
st.set_page_config(page_title="Reference to RIS Converter", page_icon="📚", layout="wide")

st.title("📚 Reference Extractor & DOI Finder (Streamlit Compatible)")
st.write("Upload a PDF *or* paste references. The app finds DOIs and exports RIS.")

input_method = st.radio("Choose input method:", ["📄 Upload PDF", "✍️ Paste References"], horizontal=True)

raw_text = ""


# -----------------------------------------------------------
#            PDF UPLOAD MODE
# -----------------------------------------------------------
if input_method == "📄 Upload PDF":
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if pdf_file:
        with st.spinner("Extracting text from PDF..."):
            raw_text = extract_text_from_pdf(pdf_file)

        if raw_text.startswith("ERROR_READING_PDF"):
            st.error(raw_text)
        else:
            st.success("PDF extracted successfully!")
            st.text_area("Extracted Text", raw_text, height=200)


# -----------------------------------------------------------
#             PASTE TEXT MODE
# -----------------------------------------------------------
else:
    raw_text = st.text_area(
        "Paste references (numbered or not):",
        height=300,
        placeholder="Paste references here..."
    )


# -----------------------------------------------------------
#                   PROCESS BUTTON
# -----------------------------------------------------------
if st.button("Convert to RIS", type="primary"):
    if not raw_text.strip():
        st.warning("Paste or upload something first.")
        st.stop()

    # Split references
    refs = split_references_auto(raw_text)

    if len(refs) == 0:
        st.error("No valid references detected.")
        st.stop()

    st.success(f"Detected {len(refs)} reference(s). Searching DOIs...")

    ris_output = ""
    results_table = []

    progress = st.progress(0)
    status = st.empty()

    for i, ref in enumerate(refs):
        status.text(f"Searching DOI for reference {i+1}/{len(refs)} ...")

        item, doi = crossref_lookup(ref)

        if doi:
            ris_output += item_to_ris(item)
            results_table.append((i+1, "FOUND", doi, ref[:80]))
        else:
            results_table.append((i+1, "NOT FOUND", "-", ref[:80]))

        progress.progress((i+1) / len(refs))

    progress.empty()
    status.empty()

    st.subheader("Results")
    for r in results_table:
        idx, status, doi, preview = r
        if status == "FOUND":
            st.success(f"{idx}. DOI Found: {doi}")
        else:
            st.warning(f"{idx}. No DOI → {preview}...")

    if ris_output:
        st.download_button(
            "📥 Download RIS File",
            data=ris_output,
            file_name="references.ris",
            mime="application/x-research-info-systems"
        )

        st.subheader("RIS Preview")
        st.code(ris_output, language="text")
    else:
        st.error("No RIS entries generated.")


st.caption("Built for Streamlit • Uses pypdf + CrossRef API")
