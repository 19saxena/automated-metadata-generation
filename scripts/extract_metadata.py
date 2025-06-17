import os
from app.utils import extract_text, generate_metadata

def batch_process(directory):
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'rb') as f:
            text = extract_text(f)
            metadata = generate_metadata(text)
            print(f"{filename}: {metadata}")

if __name__ == "__main__":
    batch_process("path_to_documents")
