
import re
import cv2
import easyocr
import numpy as np

from pathlib import Path
from collections import defaultdict
from ultralytics import YOLO

from owner_database import find_owner


# --------------------------------------------------
# 1. FILE PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

VIDEO_PATH = BASE_DIR / "traffic.mp4"
MODEL_PATH = BASE_DIR / "license_plate_detector.pt"
CROP_DIR = BASE_DIR / "plate_crops"

CROP_DIR.mkdir(exist_ok=True)

OCR_MIN_CONFIDENCE = 0.35
FRAME_STRIDE = 1
PADDING = 5


# --------------------------------------------------
# 2. NORMALIZE PLATE TEXT
# --------------------------------------------------

def normalize_plate(text):
    """Keep only letters and digits, in uppercase."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def looks_like_plate(text):
    """
    General screening for Indian and foreign plates.
    This is not a country-specific validity check.
    """
    plate = normalize_plate(text)

    # Reject strings that are too short or too long.
    if not 4 <= len(plate) <= 12:
        return False

    # Require at least one letter and one digit.
    has_letter = any(ch.isalpha() for ch in plate)
    has_digit = any(ch.isdigit() for ch in plate)

    return has_letter and has_digit


# --------------------------------------------------
# 3. LOAD MODELS
# --------------------------------------------------

print("Loading number-plate detector...")
model = YOLO(str(MODEL_PATH))

print("Loading EasyOCR...")
reader = easyocr.Reader(["en"], gpu=False)

print("Models loaded successfully.")


# --------------------------------------------------
# 4. PREPARE CROPS FOR OCR
# --------------------------------------------------

def prepare_variants(crop):
    """Create a few image versions to improve OCR."""
    variants = []

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    enlarged = cv2.resize(
        gray,
        None,
        fx=3,
        fy=3,
        interpolation=cv2.INTER_CUBIC
    )
    variants.append(enlarged)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    contrast = clahe.apply(enlarged)
    variants.append(contrast)

    threshold = cv2.adaptiveThreshold(
        contrast,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11
    )
    variants.append(threshold)

    return variants


# --------------------------------------------------
# 5. READ A PLATE CROP
# --------------------------------------------------

def read_plate(crop):
    """
    Return the strongest OCR candidate and confidence.
    Confidence is an OCR estimate, not a guarantee.
    """
    readings = []

    for image in prepare_variants(crop):
        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        for _, text, confidence in results:
            plate = normalize_plate(text)

            if looks_like_plate(plate):
                readings.append((plate, float(confidence)))

    if not readings:
        return None, 0.0

    # Aggregate repeated readings of the same normalized text.
    scores = defaultdict(list)

    for plate, confidence in readings:
        scores[plate].append(confidence)

    ranked = sorted(
        scores.items(),
        key=lambda item: (
            len(item[1]),
            sum(item[1]) / len(item[1])
        ),
        reverse=True
    )

    best_plate, confidences = ranked[0]
    average_confidence = sum(confidences) / len(confidences)

    return best_plate, average_confidence


# --------------------------------------------------
# 6. PROCESS THE ENTIRE VIDEO
# --------------------------------------------------

def main():
    if not VIDEO_PATH.exists():
        print(f"ERROR: Video not found: {VIDEO_PATH}")
        return

    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found: {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("ERROR: Could not open the video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\nStarting full-video processing...")
    print(f"Video: {VIDEO_PATH.name}")
    print(f"Reported frames: {total_frames}")
    print(f"Frame stride: {FRAME_STRIDE}")

    frame_number = 0
    frames_processed = 0
    detections = 0
    ocr_attempts = 0

    # Store observations for each plate candidate.
    candidate_data = defaultdict(
        lambda: {
            "confidences": [],
            "crop": None,
            "best_confidence": 0.0
        }
    )

    while True:
        success, frame = cap.read()

        if not success:
            break

        current_frame = frame_number
        frame_number += 1

        if current_frame % FRAME_STRIDE != 0:
            continue

        frames_processed += 1

        results = model.predict(
            source=frame,
            verbose=False,
            conf=0.25
        )

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                height, width = frame.shape[:2]

                x1 = max(0, x1 - PADDING)
                y1 = max(0, y1 - PADDING)
                x2 = min(width, x2 + PADDING)
                y2 = min(height, y2 + PADDING)

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                detections += 1

                plate, confidence = read_plate(crop)
                ocr_attempts += 1

                if not plate:
                    continue

                entry = candidate_data[plate]
                entry["confidences"].append(confidence)

                if confidence > entry["best_confidence"]:
                    entry["best_confidence"] = confidence
                    entry["crop"] = crop.copy()

    cap.release()

    # --------------------------------------------------
    # 7. SAVE CROPS AND CHECK DATABASE
    # --------------------------------------------------

    matched = 0
    unmatched = 0
    needs_review = 0

    print("\n" + "=" * 55)
    print("PLATE RECOGNITION AND OWNER LOOKUP")
    print("=" * 55)

    for plate, data in sorted(candidate_data.items()):
        confidence = data["best_confidence"]
        crop = data["crop"]

        print("\n---------------------------------------------")
        print(f"Recognized plate: {plate}")
        print(f"Observations: {len(data['confidences'])}")
        print(f"Best OCR confidence: {confidence:.2f}")

        if crop is not None:
            safe_name = re.sub(r"[^A-Z0-9_-]", "_", plate)
            crop_path = CROP_DIR / (
                f"{safe_name}_confidence_{confidence:.2f}.jpg"
            )
            cv2.imwrite(str(crop_path), crop)
            print(f"Saved crop: {crop_path}")

        if confidence < OCR_MIN_CONFIDENCE:
            print("Status: LOW CONFIDENCE - MANUAL REVIEW REQUIRED")
            needs_review += 1
            continue

        # Search only using the recognized text.
        # Do not automatically substitute similar-looking characters.
        try:
            owner = find_owner(plate)
        except Exception as error:
            print(f"Database lookup error: {error}")
            continue

        if owner:
            matched += 1
            print("Status: OWNER RECORD FOUND")
            print(f"Owner: {owner.get('owner_name', 'Unknown')}")
            print(f"Phone: {owner.get('phone_number', 'Unknown')}")
            print(f"Vehicle type: {owner.get('vehicle_type', 'Unknown')}")
        else:
            unmatched += 1
            print("Status: NO EXACT OWNER MATCH")
            print("Action: Verify the plate and vehicle database.")

    # --------------------------------------------------
    # 8. SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 55)
    print("FULL-VIDEO PIPELINE SUMMARY")
    print("=" * 55)
    print(f"Video frames read: {frame_number}")
    print(f"Frames processed: {frames_processed}")
    print(f"Total plate detections: {detections}")
    print(f"Total OCR attempts: {ocr_attempts}")
    print(f"Distinct OCR candidates: {len(candidate_data)}")
    print(f"Candidates matched in MongoDB: {matched}")
    print(f"Candidates not found in MongoDB: {unmatched}")
    print(f"Candidates needing manual review: {needs_review}")
    print(f"Saved crop folder: {CROP_DIR}")
    print("\nPIPELINE COMPLETED")


if __name__ == "__main__":
    main()
