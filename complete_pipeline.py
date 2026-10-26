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

OUTPUT_DIR = BASE_DIR / "output"
CROP_DIR = BASE_DIR / "plate_crops"

OUTPUT_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)

# MJPG in AVI is useful for testing locally.
# We can convert it to browser-friendly MP4 afterward.
OUTPUT_VIDEO = OUTPUT_DIR / "plate_recognition.avi"

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
    """General screening for Indian and foreign plates."""
    plate = normalize_plate(text)

    if not 4 <= len(plate) <= 12:
        return False

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
    """Create image variants to improve OCR."""
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
    """Return the strongest OCR candidate and confidence."""
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
# 6. VIDEO ANNOTATION HELPERS
# --------------------------------------------------

def draw_label(frame, x, y, lines, color):
    """Draw a readable background and multiple text lines."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 2
    line_height = 23
    padding = 6

    text_sizes = [
        cv2.getTextSize(
            line, font, font_scale, thickness
        )[0]
        for line in lines
    ]

    box_width = max(size[0] for size in text_sizes) + padding * 2
    box_height = line_height * len(lines) + padding * 2

    x = max(0, min(x, frame.shape[1] - box_width))
    y = max(box_height, min(y, frame.shape[0] - 5))

    top = y - box_height

    cv2.rectangle(
        frame,
        (x, top),
        (x + box_width, y),
        color,
        -1
    )

    for index, line in enumerate(lines):
        text_y = top + padding + (index + 1) * line_height - 5

        cv2.putText(
            frame,
            line,
            (x + padding, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )


def lookup_exact_owner(plate):
    """
    Use the existing database function.
    Never guess owners or substitute similar-looking characters.
    """
    try:
        return find_owner(plate), None
    except Exception as error:
        print(f"Database lookup error for {plate}: {error}")
        return None, str(error)


# --------------------------------------------------
# 7. PROCESS VIDEO AND ANNOTATE PLATES
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
        print("ERROR: Could not open the input video.")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 15.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # AVI + MJPG for a reliable local playback test.
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        print("ERROR: Could not create the output video.")
        print("Check that OpenCV supports the MJPG codec.")
        return

    print("\nStarting plate recognition...")
    print(f"Input video: {VIDEO_PATH.name}")
    print(f"Frames: {total_frames}")
    print(f"FPS: {fps:.2f}")
    print(f"Output video: {OUTPUT_VIDEO}")

    frame_number = 0
    frames_processed = 0
    detections = 0
    ocr_attempts = 0

    # Keep the best crop and confidence for each OCR candidate.
    candidate_data = defaultdict(
        lambda: {
            "confidences": [],
            "crop": None,
            "best_confidence": 0.0
        }
    )

    # Cache exact database lookups so the same plate is not
    # queried repeatedly on every frame.
    owner_cache = {}
    lookup_error_cache = {}

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            current_frame = frame_number
            frame_number += 1

            if current_frame % FRAME_STRIDE != 0:
                writer.write(frame)
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
                    ocr_attempts += 1

                    plate, confidence = read_plate(crop)

                    if not plate:
                        # Plate detected, but OCR could not read it.
                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 165, 255),
                            2
                        )

                        draw_label(
                            frame,
                            x1,
                            y1,
                            ["PLATE DETECTED", "OCR UNREADABLE"],
                            (0, 140, 255)
                        )

                        continue

                    entry = candidate_data[plate]
                    entry["confidences"].append(confidence)

                    if confidence > entry["best_confidence"]:
                        entry["best_confidence"] = confidence
                        entry["crop"] = crop.copy()

                    # Perform an exact lookup only when OCR confidence
                    # reaches the configured threshold.
                    if (
                        confidence >= OCR_MIN_CONFIDENCE
                        and plate not in owner_cache
                        and plate not in lookup_error_cache
                    ):
                        owner, error = lookup_exact_owner(plate)

                        if error:
                            lookup_error_cache[plate] = error
                        else:
                            owner_cache[plate] = owner

                    # Prepare the visible annotation.
                    if confidence < OCR_MIN_CONFIDENCE:
                        box_color = (0, 165, 255)
                        lines = [
                            f"Plate: {plate}",
                            f"OCR: {confidence:.2f} - REVIEW"
                        ]

                    elif plate in lookup_error_cache:
                        box_color = (0, 0, 255)
                        lines = [
                            f"Plate: {plate}",
                            "DATABASE ERROR"
                        ]

                    elif owner_cache.get(plate):
                        owner = owner_cache[plate]

                        owner_name = str(
                            owner.get("owner_name", "Unknown")
                        )

                        vehicle_type = str(
                            owner.get("vehicle_type", "Unknown")
                        )

                        box_color = (0, 180, 0)
                        lines = [
                            f"Plate: {plate} ({confidence:.2f})",
                            f"Owner: {owner_name}",
                            f"Type: {vehicle_type}"
                        ]

                    elif plate in owner_cache:
                        box_color = (0, 0, 255)
                        lines = [
                            f"Plate: {plate} ({confidence:.2f})",
                            "NO EXACT MATCH"
                        ]

                    else:
                        box_color = (0, 165, 255)
                        lines = [
                            f"Plate: {plate}",
                            "LOOKUP PENDING"
                        ]

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_color,
                        2
                    )

                    draw_label(
                        frame,
                        x1,
                        y1,
                        lines,
                        box_color
                    )

            # Add a small project title to every frame.
            cv2.rectangle(
                frame,
                (0, 0),
                (width, 38),
                (35, 35, 35),
                -1
            )

            cv2.putText(
                frame,
                "TrafficGuard | Number Plate Recognition",
                (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)

            if frame_number % 30 == 0:
                print(
                    f"Processed {frame_number}/{total_frames} frames"
                )

    finally:
        cap.release()
        writer.release()

    # --------------------------------------------------
    # 8. SAVE CROPS AND PRINT OWNER RESULTS
    # --------------------------------------------------

    matched = 0
    unmatched = 0
    needs_review = 0
    lookup_errors = 0

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

        if plate in lookup_error_cache:
            print("Status: DATABASE LOOKUP ERROR")
            lookup_errors += 1
            continue

        owner = owner_cache.get(plate)

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
    # 9. SUMMARY
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
    print(f"Database lookup errors: {lookup_errors}")
    print(f"Saved crop folder: {CROP_DIR}")
    print(f"Annotated video: {OUTPUT_VIDEO}")
    print("\nPIPELINE COMPLETED")


if __name__ == "__main__":
    main()
