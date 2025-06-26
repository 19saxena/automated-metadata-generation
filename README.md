#  Automated Metadata Generator
# Demo Video Link: https://drive.google.com/file/d/1iJFokYGgd3hGm0o_j7udVbbQPL_zOgM0/view?usp=drive_link

This project enables intelligent metadata extraction from various document types including PDFs, DOCX, TXT, and images using a **Flask + Streamlit** app and a powerful backend powered by NLP models.

---

## Features

-  **Text extraction** from PDF, DOCX, TXT, and images (with OCR fallback)
-  **Summarization** using BART model (optimized for CPU)
-  **Keyword extraction** using NLTK filtering
-  **Entity extraction** via spaCy
-  **Title generation** using T5
-  **Statistics** like word count, most common words, etc.

---

## Main Folder Structure

automated-metadata-generation/
│
├── app/
│ ├── utils.py # Metadata processing logic
│ ├── routes.py # Flask routes
│ ├── streamlit_app.py # Streamlit frontend
│ ├── server.py # Flask app launcher
│ └── templates/ (if used)
│
├── uploads/ # Uploaded documents (auto-created)
├── requirements.txt
└── README.md

yaml
Copy
Edit

---

##  Setup Instructions

### 1. Clone the Repository

git clone https://github.com/19saxena/automated-metadata-generation.git
cd automated-metadata-generation
2. Create a Virtual Environment and Install Dependencies

python -m venv venv
source venv/bin/activate         # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Or manually install:

pip install transformers torch pytesseract spacy nltk pillow pymupdf python-docx streamlit
python -m nltk.downloader stopwords punkt
python -m spacy download en_core_web_sm

Run the App Locally
Step 1: Start Flask Backend

cd app
python server.py
Make sure server.py includes:

if __name__ == "__main__":
    from routes import main
    app = create_app()
    app.register_blueprint(main)
    app.run(debug=True, use_reloader=False)
Step 2: In a separate terminal, start the Streamlit Frontend

cd app
streamlit run streamlit_app.py
### Usage
Upload a document (pdf, docx, txt, png, or jpg) from the Streamlit interface.

The file is sent to the Flask backend (/upload), which extracts metadata using utils.py.

Results are displayed on the page: title, summary, keywords, entities, statistics.
