import streamlit as st
import requests
import json

# ----------------------------
# CONFIG
# ----------------------------
HF_MODEL = "google/flan-t5-xl"

st.set_page_config(page_title="Reference Extractor", layout="wide")

st.title("📚 Reference Extractor + DOI Finder + RIS Export (Free, No Install)")

# User inputs HuggingFace token
hf_token = st.text_input("🔑 Enter your HuggingFace Token (free from https://huggingface.co/settings/tokens)", type="password")

if not hf_token:
    st.warning("Please enter your HuggingFace token to continue.")
    st.stop()

headers = {
    "Authorization": f"Bearer {hf_token}",
    "Content-Type": "application/json"
}

# ----------------------------
# FLAN-T5 Extraction Function
# ----------------------------
def extract_titles(text_block):
    prompt = f"""
Extract each reference separately and give ONLY the title for each one.

Example input:
[1] S.B. Lin, Recent advances in silicone pressure-sensitive adhesives, J. Adhes. Sci. Technol. 21 (2007) 605–623.

Output:
Recent advances in silicone pressure-sensitive adhesives

Now process this:
{text_block}
"""

    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 200}
    }

    response = requests.post(
        f"https://api-inference.huggingface.co/models/{HF_MODEL}",
        headers=headers,
        data=json.dumps(data)
    )

    try:
        return response.json()[0]["generated_text"].strip()
    except:
        return "ERROR parsing with HuggingFace model."


# ----------------------------
# PubMed Search
# ----------------------------
def pubmed_search(title):
    url = f"https://pubmed.ncbi.nlm.nih.gov/api/v1/publications/?term={title}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            return data
    except:
        pass
    return None


# ----------------------------
# CrossRef Search
# ----------------------------
def crossref_search(title):
    url = f"https://api.crossref.org/works?query.title={title}"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            if "items" in data["message"]:
                return data["message"]["items"]
    except:
        pass
    return None


# ----------------------------
# RIS EXPORT
# ----------------------------
def to_ris(item):
    ris = ""
    ris += "TY  - JOUR\n"
    if "title" in item:
        ris += f"TI  - {item['title'][0]}\n"
    if "author" in item:
        for a in item["author"]:
            family = a.get("family", "")
            given = a.get("given", "")
            ris += f"AU  - {family}, {given}\n"
    if "container-title" in item:
        ris += f"JO  - {item['container-title'][0]}\n"
    if "issued" in item and "date-parts" in item["issued"]:
        ris += f"PY  - {item['issued']['date-parts'][0][0]}\n"
    if "DOI" in item:
        ris += f"DO  - {item['DOI']}\n"
    ris += "ER  - \n"
    return ris


# ----------------------------
# UI Input Box
# ----------------------------
st.subheader("📥 Paste your references below")
text_block = st.text_area("Example: [1] S.B. Lin …", height=250)

if st.button("Extract Titles + Search DOIs"):
    if not text_block.strip():
        st.warning("Please paste references!")
        st.stop()

    st.info("⏳ Extracting titles using FLAN-T5-XL (HuggingFace)...")
    titles_raw = extract_titles(text_block)

    st.subheader("📝 Extracted Titles")
    st.code(titles_raw)

    # Split by newlines
    titles = [t.strip() for t in titles_raw.split("\n") if t.strip()]

    results = []
    for t in titles:
        st.write(f"### 🔍 Searching: **{t}**")

        # CrossRef
        cr = crossref_search(t)
        if cr:
            st.success("CrossRef Match Found")
            st.json(cr[0])
            results.append(cr[0])
        else:
            st.warning("No CrossRef match")

    # RIS EXPORT
    if results:
        ris_text = ""
        for r in results:
            ris_text += to_ris(r) + "\n"

        st.subheader("📄 RIS Export")
        st.code(ris_text)

        st.download_button(
            "⬇ Download RIS File",
            data=ris_text,
            file_name="references.ris",
            mime="application/x-research-info-systems"
        )
