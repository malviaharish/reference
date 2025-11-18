import re
import requests
import streamlit as st
from PyPDF2 import PdfReader
import io


def extract_text_from_pdf(pdf_file):
    """
    Extracts text from an uploaded PDF file.
    Returns the full text content as a string.
    """
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"


def detect_reference_style(text):
    """
    Attempts to detect the citation style used in the references.
    Returns a tuple of (detected_style, confidence_level)
    """
    patterns = {
        'APA': {
            'author_year': r'\b[A-Z][a-z]+,\s+[A-Z]\.\s*(?:&|and)\s+[A-Z][a-z]+,\s+[A-Z]\.\s*\(\d{4}\)',
            'journal_style': r'\.\s+[A-Z][^.]+\.\s+\d+\(\d+\)',
        },
        'MLA': {
            'author_title': r'\b[A-Z][a-z]+,\s+[A-Z][a-z]+\.\s+"[^"]+\."',
            'title_quotes': r'"[^"]+\."',
        },
        'Chicago': {
            'footnote_style': r'\d+\.\s+[A-Z][a-z]+,\s+[A-Z][a-z]+',
            'publisher': r':\s+[A-Z][^,]+,\s+\d{4}',
        },
        'IEEE': {
            'bracket_number': r'\[\d+\]\s+[A-Z]',
            'et_al': r'et\s+al\.',
        }
    }
    
    style_scores = {}
    
    for style, style_patterns in patterns.items():
        score = 0
        for pattern_name, pattern in style_patterns.items():
            matches = re.findall(pattern, text)
            score += len(matches)
        style_scores[style] = score
    
    if max(style_scores.values()) == 0:
        return "Unknown", "Low"
    
    detected_style = max(style_scores, key=style_scores.get)
    max_score = style_scores[detected_style]
    
    if max_score >= 3:
        confidence = "High"
    elif max_score >= 1:
        confidence = "Medium"
    else:
        confidence = "Low"
    
    return detected_style, confidence


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


def validate_reference(ref_text):
    """
    Validates a reference text for basic completeness.
    Returns (is_valid, issues_list)
    """
    issues = []
    
    if len(ref_text) < 20:
        issues.append("Reference seems too short")
    
    if not any(char.isdigit() for char in ref_text):
        issues.append("No year/date detected")
    
    if ref_text.count(',') < 1:
        issues.append("Missing commas (incomplete citation?)")
    
    return len(issues) == 0, issues


def detect_duplicates(results):
    """
    Detects duplicate references based on DOI.
    Returns dict mapping index to list of duplicate indices.
    """
    duplicates = {}
    doi_to_indices = {}
    
    for idx, result in enumerate(results):
        if result['found'] and result['doi']:
            doi = result['doi']
            if doi in doi_to_indices:
                if doi not in duplicates:
                    duplicates[doi] = [doi_to_indices[doi]]
                duplicates[doi].append(idx)
            else:
                doi_to_indices[doi] = idx
    
    return duplicates


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


def convert_to_bibtex(item):
    if not item:
        return ""
    
    authors = item.get("author", [])
    author_list = []
    for a in authors:
        family = a.get("family", "")
        given = a.get("given", "")
        author_list.append(f"{given} {family}".strip())
    
    year = ""
    if "issued" in item and "date-parts" in item["issued"]:
        year = str(item["issued"]["date-parts"][0][0])
    
    title = item.get('title', [''])[0]
    journal = item.get('container-title', [''])[0]
    doi = item.get('DOI', '')
    
    cite_key = f"{authors[0].get('family', 'Unknown')}{year}" if authors else f"Unknown{year}"
    
    bibtex = f"""@article{{{cite_key},
  author = {{{" and ".join(author_list)}}},
  title = {{{title}}},
  journal = {{{journal}}},
  year = {{{year}}},
  doi = {{{doi}}}
}}

"""
    return bibtex


def convert_to_endnote(item):
    if not item:
        return ""
    
    authors = item.get("author", [])
    author_lines = ""
    for a in authors:
        family = a.get("family", "")
        given = a.get("given", "")
        author_lines += f"%A {given} {family}\n".strip() + "\n"
    
    year = ""
    if "issued" in item and "date-parts" in item["issued"]:
        year = str(item["issued"]["date-parts"][0][0])
    
    title = item.get('title', [''])[0]
    journal = item.get('container-title', [''])[0]
    doi = item.get('DOI', '')
    
    endnote = f"""%0 Journal Article
%T {title}
{author_lines}%D {year}
%J {journal}
%R {doi}

"""
    return endnote


def convert_to_apa(item):
    if not item:
        return ""
    
    authors = item.get("author", [])
    if len(authors) == 0:
        author_str = "Unknown Author"
    elif len(authors) == 1:
        a = authors[0]
        author_str = f"{a.get('family', '')}, {a.get('given', '')}"
    elif len(authors) == 2:
        a1 = authors[0]
        a2 = authors[1]
        author_str = f"{a1.get('family', '')}, {a1.get('given', '')}, & {a2.get('family', '')}, {a2.get('given', '')}"
    else:
        author_list = []
        for i, a in enumerate(authors[:-1]):
            author_list.append(f"{a.get('family', '')}, {a.get('given', '')}")
        last = authors[-1]
        author_str = ", ".join(author_list) + f", & {last.get('family', '')}, {last.get('given', '')}"
    
    year = ""
    if "issued" in item and "date-parts" in item["issued"]:
        year = str(item["issued"]["date-parts"][0][0])
    
    title = item.get('title', [''])[0]
    journal = item.get('container-title', [''])[0]
    doi = item.get('DOI', '')
    
    apa = f"{author_str} ({year}). {title}. {journal}. https://doi.org/{doi}\n\n"
    return apa


st.set_page_config(page_title="Reference Converter", page_icon="📚", layout="wide")

st.title("📚 Academic Reference Converter")
st.markdown("Convert your numbered academic references into multiple citation formats using the Crossref API")

input_method = st.radio(
    "Choose input method:",
    options=["📄 Upload PDF", "✍️ Paste Text"],
    horizontal=True
)

reference_text = ""
detected_style = None
detected_confidence = None

if input_method == "📄 Upload PDF":
    st.info("**Upload a PDF** containing references. The app will automatically extract text and detect the citation style.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=['pdf'],
        help="Upload a PDF file containing academic references"
    )
    
    if uploaded_file is not None:
        with st.spinner("Extracting text from PDF..."):
            pdf_text = extract_text_from_pdf(uploaded_file)
        
        if not pdf_text.startswith("Error"):
            st.success("✅ PDF text extracted successfully!")
            
            with st.spinner("Detecting citation style..."):
                detected_style, detected_confidence = detect_reference_style(pdf_text)
            
            col_style1, col_style2 = st.columns(2)
            with col_style1:
                st.metric("Detected Citation Style", detected_style)
            with col_style2:
                confidence_color = "🟢" if detected_confidence == "High" else "🟡" if detected_confidence == "Medium" else "🔴"
                st.metric("Confidence Level", f"{confidence_color} {detected_confidence}")
            
            with st.expander("📄 View Extracted Text", expanded=False):
                st.text_area("Extracted PDF Content", pdf_text, height=200)
            
            reference_text = pdf_text
        else:
            st.error(pdf_text)

else:
    st.info("**Paste your references** below (formats like '1.' or '1)' are supported). Each reference can span multiple lines.")
    
    reference_text = st.text_area(
        "Paste your references here:",
        height=300,
        placeholder="Example:\n1. Smith, J., & Jones, A. (2020). Article title.\n   Journal Name, 15(2), 123-145.\n2. Brown, B. (2019). Another article...",
        help="Paste references exactly as copied. Multi-line references are supported."
    )
    
    if reference_text.strip():
        with st.spinner("Detecting citation style..."):
            detected_style, detected_confidence = detect_reference_style(reference_text)
        
        col_style1, col_style2 = st.columns(2)
        with col_style1:
            st.metric("Detected Citation Style", detected_style)
        with col_style2:
            confidence_color = "🟢" if detected_confidence == "High" else "🟡" if detected_confidence == "Medium" else "🔴"
            st.metric("Confidence Level", f"{confidence_color} {detected_confidence}")

export_format = st.selectbox(
    "Export Format:",
    options=["RIS", "BibTeX", "EndNote", "APA"],
    help="Choose the output format for your references"
)

if 'field_edits' not in st.session_state:
    st.session_state.field_edits = {}

if st.button("Convert References", type="primary", use_container_width=True):
    if not reference_text.strip():
        st.warning("Please paste some references first.")
    else:
        references = split_numbered_references(reference_text)
        
        if not references:
            st.warning("No numbered references detected. Make sure your references are numbered (e.g., 1., 2., etc.)")
        else:
            st.success(f"Detected {len(references)} reference(s)")
            
            validation_warnings = []
            for idx, ref in enumerate(references, 1):
                is_valid, issues = validate_reference(ref)
                if not is_valid:
                    validation_warnings.append((idx, issues))
            
            if validation_warnings:
                with st.expander("⚠️ Validation Warnings", expanded=False):
                    for ref_num, issues in validation_warnings:
                        st.warning(f"**Reference {ref_num}:** {', '.join(issues)}")
            
            format_converters = {
                "RIS": convert_to_ris,
                "BibTeX": convert_to_bibtex,
                "EndNote": convert_to_endnote,
                "APA": convert_to_apa
            }
            
            converter = format_converters[export_format]
            
            all_output = ""
            results = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, ref in enumerate(references):
                status_text.text(f"Processing reference {idx + 1} of {len(references)}...")
                
                item, doi = search_crossref(ref)
                
                if doi:
                    all_output += converter(item)
                    results.append({
                        "ref": ref,
                        "doi": doi,
                        "title": item.get('title', [''])[0] if item else "",
                        "item": item,
                        "found": True
                    })
                else:
                    results.append({
                        "ref": ref,
                        "doi": None,
                        "title": "",
                        "item": None,
                        "found": False
                    })
                
                progress_bar.progress((idx + 1) / len(references))
            
            status_text.empty()
            progress_bar.empty()
                
            st.session_state.results = results
            st.session_state.all_output = all_output
            st.session_state.export_format = export_format
            st.session_state.field_edits = {}

if 'results' in st.session_state and st.session_state.results:
    results = st.session_state.results
    all_output = st.session_state.all_output
    current_export_format = st.session_state.export_format
    
    duplicates = detect_duplicates(results)
    if duplicates:
        with st.expander("🔍 Duplicate References Detected", expanded=True):
            for doi, indices in duplicates.items():
                ref_numbers = [i + 1 for i in indices]
                st.error(f"**DOI {doi}** appears in references: {', '.join(map(str, ref_numbers))}")
                st.caption("Consider removing duplicate entries before exporting.")
    
    st.subheader("Results & Reference Editing")
    st.info("Edit any reference details below. Changes will be reflected in the export.")
    
    for idx, result in enumerate(results, 1):
        with st.expander(f"Reference {idx}: {'✅ DOI Found' if result['found'] else '❌ No DOI'}"):
            st.text(f"Original: {result['ref'][:150]}{'...' if len(result['ref']) > 150 else ''}")
            
            if result['found']:
                st.success(f"**DOI:** {result['doi']}")
                
                item = result['item']
                
                with st.form(key=f"edit_form_{idx}"):
                    st.markdown("**Edit Reference Fields:**")
                    
                    edited_title = st.text_input(
                        "Title:",
                        value=item.get('title', [''])[0],
                        key=f"title_{idx}"
                    )
                    
                    edited_journal = st.text_input(
                        "Journal:",
                        value=item.get('container-title', [''])[0],
                        key=f"journal_{idx}"
                    )
                    
                    year = ""
                    if "issued" in item and "date-parts" in item["issued"]:
                        year = str(item["issued"]["date-parts"][0][0])
                    
                    edited_year = st.text_input(
                        "Year:",
                        value=year,
                        key=f"year_{idx}"
                    )
                    
                    edited_doi = st.text_input(
                        "DOI:",
                        value=result['doi'],
                        key=f"doi_{idx}"
                    )
                    
                    submit_edits = st.form_submit_button("Save Edits")
                    
                    if submit_edits:
                        if 'field_edits' not in st.session_state:
                            st.session_state.field_edits = {}
                        st.session_state.field_edits[idx - 1] = {
                            'title': edited_title,
                            'journal': edited_journal,
                            'year': edited_year,
                            'doi': edited_doi
                        }
                        st.success("✅ Edits saved! Click 'Regenerate Export' below to apply changes.")
            else:
                st.warning("Could not find DOI for this reference")
                
                with st.form(key=f"manual_form_{idx}"):
                    st.markdown("**Manually Add Reference Details:**")
                    
                    manual_title = st.text_input(
                        "Title:",
                        key=f"manual_title_{idx}"
                    )
                    
                    manual_journal = st.text_input(
                        "Journal:",
                        key=f"manual_journal_{idx}"
                    )
                    
                    manual_year = st.text_input(
                        "Year:",
                        key=f"manual_year_{idx}"
                    )
                    
                    manual_doi = st.text_input(
                        "DOI:",
                        key=f"manual_doi_{idx}",
                        placeholder="10.1234/example.doi"
                    )
                    
                    submit_manual = st.form_submit_button("Add Manual Entry")
                    
                    if submit_manual and manual_doi:
                        if 'field_edits' not in st.session_state:
                            st.session_state.field_edits = {}
                        st.session_state.field_edits[idx - 1] = {
                            'title': manual_title,
                            'journal': manual_journal,
                            'year': manual_year,
                            'doi': manual_doi
                        }
                        st.success("✅ Manual entry saved! Click 'Regenerate Export' below to apply.")
    
    if st.session_state.field_edits:
        st.divider()
        if st.button("🔄 Regenerate Export with Edits", use_container_width=True):
            format_converters = {
                "RIS": convert_to_ris,
                "BibTeX": convert_to_bibtex,
                "EndNote": convert_to_endnote,
                "APA": convert_to_apa
            }
            
            converter = format_converters[current_export_format]
            updated_output = ""
            
            saved_edits = st.session_state.field_edits
            
            for idx, result in enumerate(results):
                if idx in saved_edits:
                    edits = saved_edits[idx]
                    
                    if result['item']:
                        updated_item = result['item'].copy()
                    else:
                        updated_item = {
                            'author': [],
                            'issued': {'date-parts': [[0]]}
                        }
                    
                    updated_item['title'] = [edits['title']]
                    updated_item['container-title'] = [edits['journal']]
                    updated_item['DOI'] = edits['doi']
                    
                    try:
                        year_int = int(edits['year']) if edits['year'] else 0
                        updated_item['issued'] = {'date-parts': [[year_int]]}
                    except ValueError:
                        updated_item['issued'] = {'date-parts': [[0]]}
                    
                    updated_output += converter(updated_item)
                else:
                    if result['item']:
                        updated_output += converter(result['item'])
            
            st.session_state.all_output = updated_output
            all_output = updated_output
            st.success("✅ Export regenerated with your edits!")
    
    if all_output:
        st.subheader(f"{current_export_format} Output")
        
        file_extensions = {
            "RIS": "ris",
            "BibTeX": "bib",
            "EndNote": "enw",
            "APA": "txt"
        }
        
        mime_types = {
            "RIS": "application/x-research-info-systems",
            "BibTeX": "application/x-bibtex",
            "EndNote": "application/x-endnote-refer",
            "APA": "text/plain"
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.download_button(
                label=f"📥 Download {current_export_format} File",
                data=all_output,
                file_name=f"references.{file_extensions[current_export_format]}",
                mime=mime_types[current_export_format],
                use_container_width=True
            )
        
        with st.expander(f"Preview {current_export_format} Format"):
            st.code(all_output, language="text")
    else:
        st.error(f"No references could be converted to {current_export_format} format.")

st.divider()
st.caption("This tool uses the Crossref API to search for DOIs and convert references to multiple citation formats.")

