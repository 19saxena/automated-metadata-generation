from flask import Blueprint, request, jsonify, current_app
from app.utils import extract_text, generate_metadata
import os,time

main = Blueprint('main', __name__)

@main.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        print("📄 Received file:", file.filename)
        # Save the uploaded file (optional)
        upload_folder = current_app.config.get('UPLOAD_FOLDER', 'uploads')
        os.makedirs(upload_folder, exist_ok=True)
        filepath = os.path.join(upload_folder, file.filename)
        file.seek(0)
        file.save(filepath)

        # Process the file for metadata
        file.seek(0)  # Reset file pointer before reading
        try:
            start = time.time()
            metadata = generate_metadata(file_input=file)
            print(f"✅ Metadata generated in {time.time() - start:.2f} seconds")

            return jsonify(metadata)
        except Exception as e:
            print("❌ ERROR during metadata generation:", e)
            return jsonify({"error": str(e)}), 500
    except Exception as e:
        print("❌ Exception in /upload route:", e)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
