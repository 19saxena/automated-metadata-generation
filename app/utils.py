import os
import re
import fitz  
import docx
import pytesseract
from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import spacy
from collections import Counter
import nltk
from summarizer import Summarizer

# Ensure NLTK stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
from nltk.corpus import stopwords

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def get_title_model():
    if not hasattr(get_title_model, "tokenizer"):
        try:
            print("📥 Loading T5 title model...")
            get_title_model.tokenizer = AutoTokenizer.from_pretrained("t5-base")
            get_title_model.model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            print("✅ T5 title model loaded.")
        except Exception as e:
            print("❌ Failed to load T5 title model:", e)
            import traceback; traceback.print_exc()
            raise
    return get_title_model.tokenizer, get_title_model.model

#############################
# Fast spaCy NER
#############################
def get_spacy_model():
    if not hasattr(get_spacy_model, "model"):
        get_spacy_model.model = spacy.load("en_core_web_sm")
    return get_spacy_model.model

#############################
# Parallel OCR for PDF pages
#############################
def ocr_page(img):
    return pytesseract.image_to_string(img)

def ocr_pdf_pages(images):
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(ocr_page, images))
    return "\n".join(texts)

#############################
# Universal Text Extraction
#############################
def extract_text(file_input, ocr_threshold=100, log_preview=True):
    filename = getattr(file_input, 'filename', None) or getattr(file_input, 'name', None)
    ext = os.path.splitext(filename or "")[-1].lower()
    text = ""

    file_data = file_input.read()
    file_input.seek(0)
    if ext == ".pdf":
        with fitz.open(stream=file_data, filetype="pdf") as doc:
            text = "\n".join(page.get_text() for page in doc)
        word_count = len(text.strip().split())
        if word_count < ocr_threshold:
            if log_preview:
                print("⚠️ PDF appears image-based — using OCR fallback...")
            images = []
            with fitz.open(stream=file_data, filetype="pdf") as doc:
                for page in doc:
                    pix = page.get_pixmap(dpi=200)  # Lower DPI for speed, still readable
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    images.append(img)
            texts = []
            if images:
                with ThreadPoolExecutor() as executor:
                    texts = list(executor.map(ocr_page, images))
                text = "\n".join(texts)
                if log_preview and texts:
                    print("🔍 OCR Preview:\n", texts[0][:500])
            else:
                text = ""
                if log_preview:
                    print("⚠️ No images found for OCR.")
    elif ext in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
        img = Image.open(BytesIO(file_data)).convert("RGB")
        text = pytesseract.image_to_string(img)
    elif ext == ".docx":
        doc = docx.Document(BytesIO(file_data))
        text = "\n".join(p.text for p in doc.paragraphs)
    elif ext == ".txt":
        text = file_data.decode("utf-8", errors="ignore")
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return text.strip()

#############################
# Text Cleaning
#############################
def clean_text(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip TOC-like or junk lines
        if re.search(r'\.{4,}|%|chapter|\.{2,}|[0-9]{1,3}\s*\.+\s*[0-9]+', line.lower()):
            continue
        if len(re.sub(r'[^\w\s]', '', line)) < 5:  # skip weird symbols
            continue
        cleaned.append(line)
    return ' '.join(cleaned)

#############################
# Summarization (Enhanced)
#############################
def split_text_for_summarization(text, max_words=600):
    words = text.split()
    return [' '.join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def get_summarizer():
    if not hasattr(get_summarizer, "model"):
        get_summarizer.model = pipeline(
            "summarization", model="facebook/bart-large-cnn", device=-1
        )
    return get_summarizer.model

from transformers import BartTokenizer, BartForConditionalGeneration
import torch

# Load once
bart_tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").to("cpu")

def split_text_by_tokens(text, max_tokens=450):
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len((current_chunk + sentence).split()) <= max_tokens:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def summarize_text(text, max_length=250, min_length=100):
    from nltk.tokenize import sent_tokenize

    chunks = split_text_by_tokens(text, max_tokens=420)
    summaries = []

    for i, chunk in enumerate(chunks[:6]):  # Cap at 6 chunks for speed
        try:
            inputs = bart_tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            summary_ids = bart_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=4,
                length_penalty=1.2,
                early_stopping=True,
                no_repeat_ngram_size=3,
            )
            summary = bart_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            summaries.append(summary.strip())
        except Exception as e:
            print(f"⚠️ Summarization failed for chunk {i}:", e)

    # Combine & clean
    final_summary = " ".join(summaries)
    sentences = sent_tokenize(final_summary)
    filtered = [s for s in sentences if len(s.split()) > 6]
    return " ".join(filtered[:12]) if filtered else "Summary could not be generated."



#############################
# Title Generation (Improved)
#############################
def generate_title(summary_text):
    """
    Generates a short academic title using T5 model.
    Falls back to first strong sentence if model fails.
    """
    tokenizer, model = get_title_model()
    from nltk.tokenize import sent_tokenize

    trimmed = " ".join(sent_tokenize(summary_text.strip())[:2])[:350]
    input_text = "generate a short academic title: " + trimmed

    try:
        input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
        output_ids = model.generate(input_ids, max_length=15, min_length=4, num_beams=5, early_stopping=True)
        title = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Heuristic fallback
        if (
            not title
            or title.lower() in trimmed.lower()
            or len(title.split()) < 3
            or len(title.split()) > 15
        ):
            print("⚠️ Fallback triggered: T5 title weak or repeating")
            title = sent_tokenize(summary_text.strip())[0]
    except Exception as e:
        print("❌ Title generation failed:", e)
        import traceback; traceback.print_exc()
        title = sent_tokenize(summary_text.strip())[0]

    return title

#############################
# Main Arguments Extraction 
#############################
def extract_main_arguments(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[:2])

#############################
# Keyword Extraction (Fast)
#############################
def extract_keywords(text, num_keywords=10):
    words = re.findall(r'\w+', text.lower())
    stop_words = set(stopwords.words('english'))
    filtered = [w for w in words if w not in stop_words and len(w) > 3]
    most_common = Counter(filtered).most_common(num_keywords)
    return [w for w, _ in most_common]

#############################
# Entity Extraction
#############################
def extract_entities(text):
    nlp = get_spacy_model()
    doc = nlp(text)
    entities = {}
    for ent in doc.ents:
        entities.setdefault(ent.label_, []).append(ent.text)
    for k in entities:
        entities[k] = list(set(entities[k]))
    return entities

#############################
# Document Statistics
#############################
def compute_statistics(text):
    words = re.findall(r'\w+', text)
    sentences = re.split(r'[.!?]+', text)
    paragraphs = text.split('\n\n')
    unique_words = set(words)
    most_common = Counter([w.lower() for w in words if w.lower() not in stopwords.words('english')]).most_common(10)
    stats = {
        "Word Count": len(words),
        "Sentence Count": len(sentences),
        "Paragraph Count": len(paragraphs),
        "Unique Words": len(unique_words),
        "Most Common Words": most_common,
        "Estimated Reading Time (mins)": max(1, len(words)//200)
    }
    return stats

#############################
# Main Metadata Function
#############################
def generate_metadata(file_input):
    try:
        text = extract_text(file_input)
        if not text or len(text.strip()) < 50:
            return {
                "📌 Title": "N/A",
                "📄 Summary": "N/A",
                "🔑 Main Arguments": "N/A",
                "🏷️ Keywords": [],
                "🧠 Entities": {},
                "📊 Statistics": {},
            }
        text = clean_text(text)
        summary = summarize_text(text)
        title = generate_title(summary)
        main_args = extract_main_arguments(text)
        keywords = extract_keywords(text)
        entities = extract_entities(text)
        stats = compute_statistics(text)
        return {
            "📌 Title": title,
            "📄 Summary": summary,
            "🔑 Main Arguments": main_args,
            "🏷️ Keywords": keywords,
            "🧠 Entities": entities,
            "📊 Statistics": stats,
        }
    except Exception as e:
        print("❌ ERROR during metadata generation:", e)
        import traceback; traceback.print_exc()
        return {"error": str(e)}
