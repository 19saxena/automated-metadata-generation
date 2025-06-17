import io
from app.utils import extract_text, generate_metadata

def test_extract_text_txt():
    file = io.BytesIO(b"Hello World\nThis is a test.")
    file.filename = "test.txt"
    text = extract_text(file)
    assert "Hello World" in text

def test_generate_metadata():
    text = "Sample Title\nThis is a sample document for testing."
    metadata = generate_metadata(text)
    assert "title" in metadata
    assert metadata["title"].startswith("Sample Title")
