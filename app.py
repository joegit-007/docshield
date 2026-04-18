import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
import io
from detector import detect_forgery

st.set_page_config(page_title="DocShield - Forgery Detector", page_icon="🔍", layout="wide")

st.title("🔍 DocShield — AI Document Forgery Detector")
st.markdown("*Upload a document to check if it's genuine or manipulated*")

# Language note
st.info("✅ Supports English, Tamil (தமிழ்), and Hindi (हिंदी) documents")

uploaded_file = st.file_uploader(
    "Upload a document (JPG, PNG, PDF)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    # Save file temporarily
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    tmp.write(uploaded_file.read())
    tmp.close()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Original Document")
        st.image(tmp.name, use_column_width=True)
    
    with st.spinner("🔎 Analyzing document..."):
        results = detect_forgery(tmp.name)
    
    with col2:
        st.subheader("🔬 ELA Analysis (tampered areas glow bright)")
        ela_colored = cv2.applyColorMap(
            cv2.cvtColor(results["ela_image"], cv2.COLOR_RGB2GRAY),
            cv2.COLORMAP_HOT
        )
        st.image(ela_colored, use_column_width=True, channels="BGR")
    
    # Verdict banner
    st.markdown("---")
    st.subheader("📋 Verdict")
    
    score = results["final_score"]
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Overall Risk Score", f"{score}%", help="Higher = more likely forged")
    col_b.metric("Image Anomaly", f"{results['ela_score']}%")
    col_c.metric("Text Inconsistency", f"{results['text_score']}%")
    
    st.markdown(f"## {results['verdict']}")
    st.progress(int(score))
    
    # Explainable reasons
    st.subheader("🧠 Why was this flagged?")
    for reason in results["reasons"]:
        st.markdown(f"- {reason}")
    
    # Highlight suspicious regions
    if results["suspicious_boxes"]:
        st.subheader("📍 Suspicious regions highlighted")
        img = Image.open(tmp.name).convert("RGB")
        draw = ImageDraw.Draw(img)
        for box in results["suspicious_boxes"]:
            pts = [(int(p[0]), int(p[1])) for p in box]
            draw.polygon(pts, outline="red")
            draw.polygon(pts, fill=(255, 0, 0, 80))
        st.image(img, use_column_width=True)
    
    os.unlink(tmp.name)

st.markdown("---")
st.caption("Built for Hackathon — Track C: Explainable AI for Document Forgery Detection")
