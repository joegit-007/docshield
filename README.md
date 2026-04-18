# DocShield — AI Document Forgery Detector

## Problem
Forged documents in college admissions and government verification.

## Solution
AI system using Error Level Analysis (ELA) + OCR text consistency 
checking to flag manipulated documents with explainable reports.

## How to run
pip install -r requirements.txt
streamlit run app.py

## Tech stack
- EasyOCR (regional language support)
- OpenCV + Scikit-image (ELA analysis)
- Streamlit (UI)
- Python
