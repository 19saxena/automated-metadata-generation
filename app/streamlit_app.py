import streamlit as st
import requests

st.set_page_config(page_title="📄 Metadata Generator")
st.title("📄 Upload Document to Generate Metadata")

uploaded_file = st.file_uploader("Upload file", type=["pdf", "docx", "txt", "jpg", "png"])

if uploaded_file is not None:
    st.write("📤 Sending file to backend...")
    with st.spinner("Processing (this might take 10–30 seconds)..."):
        backend_url = "http://127.0.0.1:5000/upload" 
        response = requests.post(
            backend_url,
            files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type or "application/octet-stream")},
        )

    if response.status_code == 200:
        result = response.json()

        st.subheader("📌 Title")
        st.write(result.get("📌 Title", "N/A"))

        st.subheader("📄 Summary")
        st.markdown(result.get("📄 Summary", "N/A"))

        st.subheader("🏷 Keywords")
        st.write(", ".join(result.get("🏷️ Keywords", [])))

        st.subheader("🧠 Entities")
        for k, v in result.get("🧠 Entities", {}).items():
            st.write(f"{k}: {', '.join(v)}")
    else:
        st.text(response.text)
        st.error("❌ Failed to fetch metadata. Please try again.")
