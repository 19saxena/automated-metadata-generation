from flask import Blueprint, request, jsonify, current_app
from .utils import extract_text, generate_metadata
import os
main = Blueprint('main', __name__)

@main.route('/upload', methods=['POST'])
def upload():
    file = request.files['file']
    # Save the uploaded file to the uploads directory
    upload_folder = current_app.config['UPLOAD_FOLDER']
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)
    filepath = os.path.join(upload_folder, file.filename)
    file.save(filepath)
    
    # Open and process the saved file
    with open(filepath, "rb") as f:
        text = extract_text(f)
        metadata = generate_metadata(text)
    return jsonify(metadata)