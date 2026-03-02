import fitz  # PyMuPDF
import docx
import io


def extract_text_from_pdf(pdf_file):
    text = ""
    # Open from in-memory file object
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text


def extract_text_from_docx(docx_file):
    # Open using in-memory file-like object
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text
