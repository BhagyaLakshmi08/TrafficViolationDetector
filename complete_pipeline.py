import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import Counter

from owner_database import find_owner
from fine_calculator import calculate_fine


print("========================================")
print("TRAFFIC VIOLATION COMPLETE PIPELINE")
print("========================================")


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

print("Loading vehicle/plate detector...")

plate_model = YOLO("license_plate_detector.pt")

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("Models loaded successfully.")


# -------------------------------------------------
# OPEN VIDEO
# -------------------------------------------------

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():

    print("ERROR: Could not open traffic.mp4")
    exit()


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

MAX_FRAMES = 180

DETECT_EVERY_N_FRAMES = 3

MIN_PLATE_CONFIDENCE = 0.30

MIN_OCR_CONFIDENCE = 0.40

frame_count = 0

plate_readings = []


print()
print("Processing video...")
print()


# -------------------------------------------------
# PROCESS VIDEO
# -------------------------------------------------

while frame_count < MAX_FRAMES:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1


    # Process only every 3rd frame
    if frame_count % DETECT_EVERY_N_FRAMES != 0:
        continue


    print(
        "Processing frame:",
        frame_count
    )


    # -------------------------------------------------
    # LICENSE PLATE DETECTION
    # -------------------------------------------------

    results = plate_model(
        frame,
        verbose=False
    )


    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            plate_confidence = float(
                box.conf[0]
            )


            if plate_confidence < MIN_PLATE_CONFIDENCE:
                continue


            # -------------------------------------------------
            # COORDINATES
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(
                frame.shape[1],
                x2
            )

            y2 = min(
                frame.shape[0],
                y2
            )


            # -------------------------------------------------
            # PLATE CROP
            # -------------------------------------------------

            plate = frame[
                y1:y2,
                x1:x2
            ]


            if plate.size == 0:
                continue


            # Ignore extremely small plates
            if plate.shape[1] < 30:
                continue


            if plate.shape[0] < 10:
                continue


            # -------------------------------------------------
            # GRAYSCALE
            # -------------------------------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )


            # -------------------------------------------------
            # ENLARGE
            # -------------------------------------------------

            gray = cv2.resize(
                gray,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )


            # -------------------------------------------------
            # CONTRAST
            # -------------------------------------------------

            gray = cv2.equalizeHist(
                gray
            )


            # -------------------------------------------------
            # OCR
            # -------------------------------------------------

            ocr_results = reader.readtext(

                gray,

                detail=1,

                paragraph=False,

                allowlist=(
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "0123456789"
                )
            )


            # -------------------------------------------------
            # OCR RESULTS
            # -------------------------------------------------

            for detection in ocr_results:

                text = detection[1]

                confidence = float(
                    detection[2]
                )


                if confidence < MIN_OCR_CONFIDENCE:
                    continue


                cleaned = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )


                if len(cleaned) < 4:
                    continue


                if len(cleaned) > 12:
                    continue


                plate_readings.append(
                    (
                        cleaned,
                        confidence
                    )
                )


                print(
                    "   OCR:",
                    cleaned,
                    "| Confidence:",
                    round(confidence, 2)
                )


# -------------------------------------------------
# CLOSE VIDEO
# -------------------------------------------------

video.release()


# -------------------------------------------------
# OCR ANALYSIS
# -------------------------------------------------

print()
print("========================================")
print("OCR ANALYSIS")
print("========================================")


if not plate_readings:

    print("No reliable plate detected.")

    exit()


# -------------------------------------------------
# GROUP OCR RESULTS
# -------------------------------------------------

counter = Counter()

confidence_scores = {}


for plate, confidence in plate_readings:

    counter[plate] += 1

    confidence_scores.setdefault(
        plate,
        []
    )

    confidence_scores[plate].append(
        confidence
    )


# -------------------------------------------------
# DISPLAY
# -------------------------------------------------

print()

for plate, count in counter.most_common():

    avg_confidence = (
        sum(confidence_scores[plate])
        /
        len(confidence_scores[plate])
    )

    print(
        plate,
        "->",
        count,
        "readings | Average confidence:",
        round(avg_confidence, 2)
    )


# -------------------------------------------------
# SELECT BEST PLATE
# -------------------------------------------------

best_plate = None

best_score = -1


for plate in counter:

    count = counter[plate]

    avg_confidence = (
        sum(confidence_scores[plate])
        /
        len(confidence_scores[plate])
    )


    score = count * avg_confidence


    if score > best_score:

        best_score = score

        best_plate = plate


# -------------------------------------------------
# FINAL PLATE
# -------------------------------------------------

print()
print("========================================")
print("FINAL OCR PLATE")
print("========================================")

print(
    "Plate:",
    best_plate
)

print(
    "Score:",
    round(best_score, 2)
)


# -------------------------------------------------
# MONGODB OWNER LOOKUP
# -------------------------------------------------

print()
print("========================================")
print("MONGODB OWNER LOOKUP")
print("========================================")


owner = find_owner(
    best_plate
)


if not owner:

    print(
        "Owner not found for plate:",
        best_plate
    )

    print()
    print(
        "OCR worked, but this plate is not "
        "present in the owner database."
    )

    exit()


print(
    "Owner:",
    owner["owner_name"]
)

print(
    "Phone:",
    owner["phone_number"]
)


# -------------------------------------------------
# VIOLATION
# -------------------------------------------------

# For testing we use red_light.
# Later this will come from the actual
# violation detection system.

violation_type = "red_light"


print()
print("Violation:", violation_type)


# -------------------------------------------------
# CALCULATE FINE
# -------------------------------------------------

fine = calculate_fine(
    violation_type
)


print(
    "Fine: ₹",
    fine
)


# -------------------------------------------------
# CREATE SMS MESSAGE
# -------------------------------------------------

message = (
    "Traffic Violation Alert!\n"
    f"Vehicle: {best_plate}\n"
    f"Owner: {owner['owner_name']}\n"
    f"Violation: {violation_type}\n"
    f"Fine: ₹{fine}\n"
    "Please follow traffic rules."
)


# -------------------------------------------------
# FINAL RESULT
# -------------------------------------------------

print()
print("========================================")
print("COMPLETE PIPELINE RESULT")
print("========================================")

print(
    "Vehicle:",
    best_plate
)

print(
    "Owner:",
    owner["owner_name"]
)

print(
    "Phone:",
    owner["phone_number"]
)

print(
    "Violation:",
    violation_type
)

print(
    "Fine: ₹",
    fine
)

print()
print("SMS MESSAGE:")
print(message)

print()
print("========================================")
print("PIPELINE COMPLETED")
print("========================================")