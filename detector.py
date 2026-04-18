import cv2
import numpy as np
from PIL import Image
from skimage import filters, measure
import easyocr
import re

# Initialize OCR (supports English + regional scripts like Tamil, Hindi)
reader_ta = easyocr.Reader(['en', 'ta'], gpu=False)
reader_hi = easyocr.Reader(['en', 'hi'], gpu=False)

def run_ela(image_path, quality=90):
    """Error Level Analysis - detects image editing artifacts"""
    original = Image.open(image_path).convert('RGB')
    
    # Save at lower quality and compare
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    original.save(tmp.name, 'JPEG', quality=quality)
    compressed = Image.open(tmp.name)
    
    # Calculate pixel difference
    ela_image = np.array(original, dtype=np.float32) - np.array(compressed, dtype=np.float32)
    ela_image = np.abs(ela_image) * 15  # amplify differences
    ela_image = np.clip(ela_image, 0, 255).astype(np.uint8)
    
    os.unlink(tmp.name)
    return ela_image

def analyze_text_consistency(image_path):
    """Check for font/layout inconsistencies using OCR"""
    results_ta = reader_ta.readtext(image_path, detail=1)
    results_hi = reader_hi.readtext(image_path, detail=1)

results = results_ta + results_hi
    
    if not results:
        return [], 0.0, "No text detected"
    
    # Check font size consistency (bounding box heights)
    heights = []
    for (bbox, text, conf) in results:
        top_left, top_right, bottom_right, bottom_left = bbox
        height = abs(bottom_right[1] - top_right[1])
        heights.append(height)
    
    suspicious_regions = []
    suspicion_score = 0.0
    reasons = []
    
    if heights:
        mean_h = np.mean(heights)
        std_h = np.std(heights)
        
        # Flag text regions with unusual sizes
        for i, (bbox, text, conf) in enumerate(results):
            h = heights[i]
            if abs(h - mean_h) > 2 * std_h and std_h > 0:
                suspicious_regions.append(bbox)
                reasons.append(f"Unusual text size near: '{text[:30]}'")
                suspicion_score += 0.3
        
        # Check for very low OCR confidence (blurred/manipulated text)
        low_conf = [(bbox, text) for (bbox, text, conf) in results if conf < 0.5]
        if low_conf:
            suspicion_score += 0.2 * len(low_conf)
            reasons.append(f"{len(low_conf)} region(s) have blurry/unclear text")
    
    suspicion_score = min(suspicion_score, 1.0)
    return suspicious_regions, suspicion_score, "; ".join(reasons) if reasons else "Text looks consistent"

def detect_forgery(image_path):
    """Main detection function - returns full analysis"""
    results = {}
    
    # 1. ELA Analysis
    ela = run_ela(image_path)
    ela_mean = np.mean(ela)
    ela_score = min(ela_mean / 50.0, 1.0)  # normalize 0-1
    
    # 2. Text Analysis
    suspicious_boxes, text_score, text_reason = analyze_text_consistency(image_path)
    
    # 3. Combined confidence score
    final_score = (ela_score * 0.5) + (text_score * 0.5)
    
    # 4. Decision
    if final_score > 0.6:
        verdict = "🔴 Likely FORGED"
        confidence = "High"
    elif final_score > 0.35:
        verdict = "🟡 Suspicious"
        confidence = "Medium"
    else:
        verdict = "🟢 Likely GENUINE"
        confidence = "Low risk"
    
    results = {
        "verdict": verdict,
        "confidence": confidence,
        "final_score": round(final_score * 100, 1),
        "ela_score": round(ela_score * 100, 1),
        "text_score": round(text_score * 100, 1),
        "reasons": [
            f"Image-level anomaly score: {round(ela_score*100,1)}%",
            f"Text inconsistency: {text_reason}",
        ],
        "suspicious_boxes": suspicious_boxes,
        "ela_image": ela,
    }
    return results
