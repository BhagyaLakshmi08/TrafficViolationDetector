import cv2
from ultralytics import YOLO
import easyocr

print("Starting License Plate OCR...")

# Load license plate model
plate_model = YOLO("license_plate_detector.pt")

# Load EasyOCR
print("Loading EasyOCR...")
reader = easyocr.Reader(['en'])

print("EasyOCR loaded successfully.")

# Open traffic video
video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()

# Get video properties
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

# Create output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    "plate_ocr_output.mp4",
    fourcc,
    fps,
    (width, height)
)

print("Traffic video opened successfully.")
print("License plate detection + OCR started.")

while True:

    ret, frame = video.read()

    if not ret:
        print("Video finished.")
        break

    # Detect license plates
    results = plate_model(frame, verbose=False)

    for result in results:

        for box in result.boxes:

            # Detection confidence
            confidence = float(box.conf[0])

            if confidence < 0.30:
                continue

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)

            # Crop plate
            plate = frame[y1:y2, x1:x2]

            if plate.size == 0:
                continue

            # Convert to grayscale
            gray = cv2.cvtColor(
                plate,
                cv2.COLOR_BGR2GRAY
            )

            # Enlarge plate
            gray = cv2.resize(
                gray,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )

            # OCR
            ocr_results = reader.readtext(gray)

            plate_text = ""

            for detection in ocr_results:

                text = detection[1]
                ocr_confidence = detection[2]

                if ocr_confidence >= 0.30:
                    plate_text += text + " "

            plate_text = plate_text.strip()

            # Draw plate rectangle
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )

            # Display OCR result on frame
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

    # Save processed frame
    output.write(frame)

print("Saving completed video...")

video.release()
output.release()

print("OCR processing finished.")
print("Output saved as: plate_ocr_output.mp4")