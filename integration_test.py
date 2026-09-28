import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import Counter

from violation_manager import process_violation


print("Starting quick OCR integration test...")

# -----------------------------
# LOAD MODELS
# -----------------------------

plate_model = YOLO("license_plate_detector.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])
print("EasyOCR loaded successfully.")


# -----------------------------
# OPEN VIDEO
# -----------------------------

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()


# -----------------------------
# SETTINGS
# -----------------------------

MAX_FRAMES = 100
frame_count = 0

plate_readings = []


# -----------------------------
# PROCESS FIRST 100 FRAMES
# -----------------------------

while frame_count < MAX_FRAMES:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1

    print("Processing frame:", frame_count)


    # -----------------------------
    # DETECT LICENSE PLATE
    # -----------------------------

    results = plate_model(
        frame,
        verbose=False
    )


    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            if confidence < 0.30:
                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue


            # -----------------------------
            # PREPROCESS
            # -----------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            gray = cv2.resize(
                gray,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )


            # -----------------------------
            # OCR
            # -----------------------------

            ocr_results = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )


            for detection in ocr_results:

                text = detection[1]
                ocr_confidence = detection[2]

                if ocr_confidence < 0.30:
                    continue


                cleaned = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )


                if len(cleaned) >= 3:

                    plate_readings.append(cleaned)

                    print(
                        "OCR detected:",
                        cleaned,
                        "Confidence:",
                        round(ocr_confidence, 2)
                    )


# -----------------------------
# CLOSE VIDEO
# -----------------------------

video.release()


# -----------------------------
# OCR RESULT
# -----------------------------

print()
print("--------------------------------")
print("QUICK OCR TEST FINISHED")
print("--------------------------------")


if not plate_readings:

    print("No readable plate found.")
    exit()


counts = Counter(plate_readings)

print()
print("OCR readings:")

for text, count in counts.most_common():

    print(
        text,
        "->",
        count,
        "times"
    )


final_plate = counts.most_common(1)[0][0]


print()
print("FINAL TEST PLATE:")
print(final_plate)


# -----------------------------
# CONNECT TO VIOLATION MANAGER
# -----------------------------

print()
print("--------------------------------")
print("TESTING VIOLATION SYSTEM")
print("--------------------------------")


# Temporary test violation
violation_type = "red_light"


result = process_violation(
    final_plate,
    violation_type
)


# -----------------------------
# FINAL RESULT
# -----------------------------

print()

if result:

    print("SUCCESS!")
    print("OCR → Owner → Fine connection works.")

else:

    print("OCR worked, but this plate is not")
    print("present in the owner database.")