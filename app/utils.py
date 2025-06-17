import docx
import pytesseract
from pdfminer.high_level import extract_text as pdf_extract

def extract_text(file):
    filename = file.filename
    if filename.endswith('.pdf'):
        return pdf_extract(file)
    elif filename.endswith('.docx'):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])
    elif filename.endswith('.txt'):
        return file.read().decode('utf-8')
    else:
        return pytesseract.image_to_string(file)

def generate_metadata(text):
    # Placeholder: Replace with NLP logic
    return {
        "title": text.split('\n')[0][:50],
        "summary": text[:200]
    }
