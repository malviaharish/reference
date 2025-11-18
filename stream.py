import re
import requests
import streamlit as st

def split_numbered_references(text):
    """
    Takes raw pasted reference text and splits it into complete reference blocks.
    Works for formats like:
    1. First line
       continuation line...
    2. Next reference...
    """
    lines = text.splitlines()
    refs = []
    current = []

    for line in lines:
        if re.match(r"^\s*\d+[\.\)]\s*", line):
            if current:
                refs.append(" ".join(current).strip())
                current = []
            line = re.sub(r"^\s*\d+[\.\)]\s*", "", line).strip()
            current.append(line)
        else:
            if line.strip():
                current.append(line.strip())

    if current:
        refs.append(" ".join(current).strip())

    return refs


def search_crossref(ref_text):
    url = "https://api.crossref.org/works"
    params = {"query": ref_text, "rows": 1}

    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()

        items = data["message"].get("items", [])
        if items:
            item = items[0]
            doi = item.get("DOI", "")
            return item, doi
    except:
        return None, ""

    return None, ""


def convert_to_ris(item):
    if not item:
        return ""

    authors = item.get("author", [])
    au_lines = ""

    for a in authors:
        family = a.get("family", "")
        given = a.get("given", "")
        name = f"{family}, {given}".strip().strip(",")
        au_lines += f"AU  - {name}\n"

    year = ""
    if "issued" in item and "date-parts" in item["issued"]:
        year = item["issued"]["date-parts"][0][0]

    ris = (
        "TY  - JOUR\n"
        f"TI  - {item.get('title', [''])[0]}\n"
        f"JO  - {item.get('container-title', [''])[0]}\n"
        f"PY  - {year}\n"
        f"{au_lines}"
        f"DO  - {item.get('DOI', '')}\n"
        "ER  - \n\n"
    )
    return ris


st.set_page_config(page_title="Reference to RIS Converter", page_icon="📚", layout="wide")

st.title("📚 Reference to RIS Converter")
st.markdown("Convert your numbered academic references into RIS format using the Crossref API")

st.info("**How to use:** Paste your numbered references below (formats like '1.' or '1)' are supported). Each reference can span multiple lines.")

reference_text = st.text_area(
    "Paste your references here:",
    height=300,
    placeholder="Example:\n1. Smith, J., & Jones, A. (2020). Article title.\n   Journal Name, 15(2), 123-145.\n2. Brown, B. (2019). Another article...",
    help="Paste references exactly as copied. Multi-line references are supported."
)

if st.button("Convert to RIS", type="primary", use_container_width=True):
    if not reference_text.strip():
        st.warning("Please paste some references first.")
    else:
        references = split_numbered_references(reference_text)
        
        if not references:
            st.warning("No numbered references detected. Make sure your references are numbered (e.g., 1., 2., etc.)")
        else:
            st.success(f"Detected {len(references)} reference(s)")
            
            all_ris = ""
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ref in enumerate(references):
                status_text.text(f"Processing reference {idx + 1} of {len(references)}...")
                
                item, doi = search_crossref(ref)
                
                if doi:
                    all_ris += convert_to_ris(item)
                    results.append({
                        "ref": ref,
                        "doi": doi,
                        "title": item.get('title', [''])[0] if item else "",
                        "found": True
                    })
                else:
                    results.append({
                        "ref": ref,
                        "doi": None,
                        "title": "",
                        "found": False
                    })
                
                progress_bar.progress((idx + 1) / len(references))
            
            status_text.empty()
            progress_bar.empty()
            
            st.subheader("Results")
            
            for idx, result in enumerate(results, 1):
                with st.expander(f"Reference {idx}: {'✅ DOI Found' if result['found'] else '❌ No DOI'}"):
                    st.text(f"Original: {result['ref'][:150]}{'...' if len(result['ref']) > 150 else ''}")
                    if result['found']:
                        st.success(f"**DOI:** {result['doi']}")
                        st.info(f"**Title:** {result['title']}")
                    else:
                        st.warning("Could not find DOI for this reference")
            
            if all_ris:
                st.subheader("RIS Output")
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.download_button(
                        label="📥 Download RIS File",
                        data=all_ris,
                        file_name="references.ris",
                        mime="application/x-research-info-systems",
                        use_container_width=True
                    )
                
                with st.expander("Preview RIS Format"):
                    st.code(all_ris, language="text")
            else:
                st.error("No references could be converted to RIS format.")

st.divider()
st.caption("This tool uses the Crossref API to search for DOIs and convert references to RIS format.")
