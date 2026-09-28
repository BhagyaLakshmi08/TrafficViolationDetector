import cv2
from ultralytics import YOLO
import easyocr
import re
from collections import defaultdict


print("========================================")
print("FAST LICENSE PLATE OCR TEST")
print("========================================")


# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

print("Loading license plate detector...")

plate_model = YOLO("license_plate_detector.pt")

print("Loading EasyOCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("EasyOCR loaded successfully.")


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

MAX_FRAMES = 150

# YOLO will only process every 3rd frame
DETECT_EVERY_N_FRAMES = 3

# OCR minimum confidence
MIN_OCR_CONFIDENCE = 0.40

# YOLO minimum plate confidence
MIN_PLATE_CONFIDENCE = 0.30


frame_count = 0

# Store OCR results
plate_scores = defaultdict(float)
plate_counts = defaultdict(int)


# -------------------------------------------------
# PROCESS VIDEO
# -------------------------------------------------

while frame_count < MAX_FRAMES:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1


    # -------------------------------------------------
    # SKIP FRAMES
    # -------------------------------------------------

    if frame_count % DETECT_EVERY_N_FRAMES != 0:
        continue


    print(
        "Processing frame:",
        frame_count
    )


    # -------------------------------------------------
    # YOLO PLATE DETECTION
    # -------------------------------------------------

    results = plate_model(
        frame,
        verbose=False
    )


    for result in results:

        for box in result.boxes:

            plate_confidence = float(
                box.conf[0]
            )


            if plate_confidence < MIN_PLATE_CONFIDENCE:
                continue


            # -------------------------------------------------
            # GET COORDINATES
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
            # CROP PLATE
            # -------------------------------------------------

            plate = frame[
                y1:y2,
                x1:x2
            ]


            if plate.size == 0:
                continue


            # -------------------------------------------------
            # CHECK PLATE SIZE
            # -------------------------------------------------

            plate_width = x2 - x1
            plate_height = y2 - y1


            if plate_width < 30:
                continue

            if plate_height < 10:
                continue


            # -------------------------------------------------
            # GRAYSCALE
            # -------------------------------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )


            # -------------------------------------------------
            # RESIZE
            # -------------------------------------------------

            gray = cv2.resize(
                gray,
                None,
                fx=5,
                fy=5,
                interpolation=cv2.INTER_CUBIC
            )


            # -------------------------------------------------
            # LIGHT CONTRAST ENHANCEMENT
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
            # PROCESS OCR RESULTS
            # -------------------------------------------------

            for detection in ocr_results:

                text = detection[1]

                confidence = float(
                    detection[2]
                )


                # Ignore weak OCR
                if confidence < MIN_OCR_CONFIDENCE:
                    continue


                # -------------------------------------------------
                # CLEAN TEXT
                # -------------------------------------------------

                cleaned = re.sub(
                    r'[^A-Z0-9]',
                    '',
                    text.upper()
                )


                # -------------------------------------------------
                # LENGTH CHECK
                # -------------------------------------------------

                if len(cleaned) < 3:
                    continue

                if len(cleaned) > 12:
                    continue


                # -------------------------------------------------
                # STORE RESULT
                # -------------------------------------------------

                plate_counts[cleaned] += 1

                plate_scores[cleaned] += confidence


                print(
                    "   OCR:",
                    cleaned,
                    "| Confidence:",
                    round(confidence, 2)
                )


# -------------------------------------------------
# RELEASE VIDEO
# -------------------------------------------------

video.release()


# -------------------------------------------------
# FINAL RESULTS
# -------------------------------------------------

print()
print("========================================")
print("FAST OCR TEST FINISHED")
print("========================================")


if not plate_counts:

    print()
    print("No reliable plate reading found.")

    print()
    print("Possible reasons:")
    print("- Plate is still too blurry")
    print("- OCR confidence is too low")
    print("- Plate characters are partially hidden")

    exit()


# -------------------------------------------------
# CALCULATE RESULTS
# -------------------------------------------------

final_results = []


for plate in plate_counts:

    count = plate_counts[plate]

    total_confidence = plate_scores[plate]

    average_confidence = (
        total_confidence / count
    )


    final_results.append(
        (
            plate,
            count,
            average_confidence,
            total_confidence
        )
    )


# -------------------------------------------------
# SORT
# -------------------------------------------------

final_results.sort(
    key=lambda x: x[3],
    reverse=True
)


# -------------------------------------------------
# DISPLAY
# -------------------------------------------------

print()
print("Reliable OCR readings:")
print()


for (
    plate,
    count,
    average_confidence,
    total_confidence
) in final_results:

    print(
        f"{plate} -> "
        f"{count} readings | "
        f"Average confidence: "
        f"{average_confidence:.2f} | "
        f"Score: "
        f"{total_confidence:.2f}"
    )


# -------------------------------------------------
# BEST PLATE
# -------------------------------------------------

best_plate = final_results[0][0]

best_count = final_results[0][1]

best_confidence = final_results[0][2]


print()
print("========================================")
print("BEST OCR RESULT")
print("========================================")

print(
    "Plate:",
    best_plate
)

print(
    "Readings:",
    best_count
)

print(
    "Average confidence:",
    round(best_confidence, 2)
)

print()
print("OCR test completed successfully.")