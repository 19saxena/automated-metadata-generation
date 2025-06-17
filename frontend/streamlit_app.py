import streamlit as st
import requests

st.title("Automated Metadata Generator")

uploaded_file = st.file_uploader("Upload Document", type=["pdf", "docx", "txt", "png", "jpg"])
if uploaded_file:
    files = {'file': uploaded_file}
    response = requests.post("http://localhost:5000/upload", files=files)
    if response.ok:
        st.json(response.json())
    else:
        st.error("Failed to extract metadata.")
