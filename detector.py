import cv2
import numpy as np
from PIL import Image
import easyocr

# OCR Readers
reader_ta = easyocr.Reader(['en', 'ta'], gpu=False)
reader_hi = easyocr.Reader(['en', 'hi'], gpu=False)

def run_ela(image_path, quality=90):
    original = Image.open(image_path).convert('RGB')

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
    original.save(tmp.name, 'JPEG', quality=quality)
    compressed = Image.open(tmp.name)

    ela_image = np.array(original, dtype=np.float32) - np.array(compressed, dtype=np.float32)
    ela_image = np.abs(ela_image) * 15
    ela_image = np.clip(ela_image, 0, 255).astype(np.uint8)

    os.unlink(tmp.name)
    return ela_image

def analyze_text_consistency(image_path):
    results_ta = reader_ta.readtext(image_path, detail=1)
    results_hi = reader_hi.readtext(image_path, detail=1)

    results = results_ta + results_hi

    if not results:
        return [], 0.0, "No text detected"

    heights = []
    for (bbox, text, conf) in results:
        height = abs(bbox[2][1] - bbox[1][1])
        heights.append(height)

    suspicious_regions = []
    suspicion_score = 0.0
    reasons = []

    if heights:
        mean_h = np.mean(heights)
        std_h = np.std(heights)

        for i, (bbox, text, conf) in enumerate(results):
            h = heights[i]
            if std_h > 0 and abs(h - mean_h) > 2 * std_h:
                suspicious_regions.append(bbox)
                reasons.append(f"Unusual text near: {text[:20]}")
                suspicion_score += 0.3

        low_conf = [(bbox, text) for (bbox, text, conf) in results if conf < 0.5]
        if low_conf:
            suspicion_score += 0.2 * len(low_conf)
            reasons.append(f"{len(low_conf)} low-confidence regions")

    suspicion_score = min(suspicion_score, 1.0)

    return suspicious_regions, suspicion_score, "; ".join(reasons)

def detect_forgery(image_path):
    ela = run_ela(image_path)
    ela_score = min(np.mean(ela) / 50.0, 1.0)

    suspicious_boxes, text_score, text_reason = analyze_text_consistency(image_path)

    final_score = (ela_score * 0.5) + (text_score * 0.5)

    if final_score > 0.6:
        verdict = "🔴 Likely FORGED"
    elif final_score > 0.35:
        verdict = "🟡 Suspicious"
    else:
        verdict = "🟢 Likely GENUINE"

    return {
        "verdict": verdict,
        "final_score": round(final_score * 100, 1),
        "ela_score": round(ela_score * 100, 1),
        "text_score": round(text_score * 100, 1),
        "reasons": [
            f"ELA score: {round(ela_score*100,1)}%",
            f"Text analysis: {text_reason}",
        ],
        "suspicious_boxes": suspicious_boxes,
        "ela_image": ela,
    }
