import cv2
from ultralytics import YOLO
import easyocr
import re

print("Starting Improved License Plate OCR...")

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------

plate_model = YOLO("license_plate_detector.pt")

print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])

print("EasyOCR loaded successfully.")


# -------------------------------------------------
# OPEN VIDEO
# -------------------------------------------------

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()


# -------------------------------------------------
# VIDEO SETTINGS
# -------------------------------------------------

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# -------------------------------------------------
# OUTPUT VIDEO
# -------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    "plate_ocr_output.mp4",
    fourcc,
    fps,
    (width, height)
)


print("Traffic video opened successfully.")
print("Improved license plate OCR started.")


# -------------------------------------------------
# PROCESS VIDEO
# -------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break


    # -------------------------------------------------
    # LICENSE PLATE DETECTION
    # -------------------------------------------------

    results = plate_model(frame, verbose=False)


    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.30:
                continue


            # -------------------------------------------------
            # GET PLATE COORDINATES
            # -------------------------------------------------

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            # -------------------------------------------------
            # CROP PLATE
            # -------------------------------------------------

            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue


            # -------------------------------------------------
            # CREATE MULTIPLE OCR VERSIONS
            # -------------------------------------------------

            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )


            # Version 1: Enlarged grayscale

            gray_large = cv2.resize(
                gray,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )


            # Version 2: Contrast enhancement

            contrast = cv2.equalizeHist(gray_large)


            # Version 3: Threshold

            _, threshold = cv2.threshold(
                contrast,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )


            # Version 4: Adaptive threshold

            adaptive = cv2.adaptiveThreshold(
                contrast,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            )


            # -------------------------------------------------
            # RUN OCR ON ALL VERSIONS
            # -------------------------------------------------

            images = [
                gray_large,
                contrast,
                threshold,
                adaptive
            ]

            possible_texts = []


            for image in images:

                ocr_results = reader.readtext(
                    image,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                )


                for detection in ocr_results:

                    text = detection[1]
                    ocr_confidence = detection[2]


                    if ocr_confidence >= 0.25:

                        # Remove unwanted characters
                        cleaned = re.sub(
                            r'[^A-Z0-9]',
                            '',
                            text.upper()
                        )


                        if len(cleaned) >= 2:

                            possible_texts.append(
                                (
                                    cleaned,
                                    ocr_confidence
                                )
                            )


            # -------------------------------------------------
            # SELECT BEST OCR RESULT
            # -------------------------------------------------

            plate_text = ""

            if possible_texts:

                possible_texts.sort(
                    key=lambda x: x[1],
                    reverse=True
                )

                plate_text = possible_texts[0][0]


            # -------------------------------------------------
            # DRAW PLATE BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )


            # -------------------------------------------------
            # DISPLAY RESULT
            # -------------------------------------------------

            if plate_text:

                label = "PLATE: " + plate_text

                print(
                    "Detected Plate:",
                    plate_text
                )

            else:

                label = "PLATE: Reading..."


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


    # -------------------------------------------------
    # SAVE FRAME
    # -------------------------------------------------

    output.write(frame)


# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

print("Saving completed video...")

video.release()
output.release()

print("OCR processing finished.")
print("Output saved as: plate_ocr_output.mp4")