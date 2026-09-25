import cv2
from ultralytics import YOLO
import easyocr
from pymongo import MongoClient
import re

print("Starting OCR + MongoDB system...")


# =====================================================
# 1. MONGODB
# =====================================================

MONGO_URI = "YOUR_WORKING_MONGODB_CONNECTION_STRING"

client = MongoClient(MONGO_URI)

db = client["TrafficViolationDetector"]

vehicles = db["vehicles"]

try:
    client.admin.command("ping")
    print("MongoDB connected successfully!")

except Exception as e:
    print("MongoDB connection failed:")
    print(e)
    exit()


# =====================================================
# 2. LICENSE PLATE MODEL
# =====================================================

print("Loading license plate model...")

plate_model = YOLO("license_plate_detector.pt")

print("License plate model loaded.")


# =====================================================
# 3. OCR
# =====================================================

print("Loading EasyOCR...")

reader = easyocr.Reader(["en"])

print("EasyOCR loaded.")


# =====================================================
# 4. VIDEO
# =====================================================

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# =====================================================
# 5. OUTPUT VIDEO
# =====================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

output = cv2.VideoWriter(
    "ocr_database_output.mp4",
    fourcc,
    fps,
    (width, height)
)


print("Traffic video opened.")
print("OCR + MongoDB processing started.")


# =====================================================
# 6. PROCESS VIDEO
# =====================================================

frame_count = 0

plates_detected = 0
ocr_detected = 0

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1


    # -------------------------------------------------
    # LICENSE PLATE DETECTION
    # -------------------------------------------------

    results = plate_model(
        frame,
        conf=0.20,
        verbose=False
    )


    for result in results:

        for box in result.boxes:

            confidence = float(box.conf[0])


            # -------------------------------------------------
            # COORDINATES
            # -------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            x1 = max(0, x1)
            y1 = max(0, y1)

            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            # -------------------------------------------------
            # PLATE CROP
            # -------------------------------------------------

            plate = frame[
                y1:y2,
                x1:x2
            ]


            if plate.size == 0:
                continue


            plates_detected += 1

            print(
                "License plate detected | "
                "Frame:",
                frame_count,
                "| Confidence:",
                round(confidence, 2)
            )


            # -------------------------------------------------
            # ENLARGE PLATE
            # -------------------------------------------------

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


            # -------------------------------------------------
            # OCR
            # -------------------------------------------------

            ocr_results = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )


            plate_text = ""

            best_confidence = 0


            for detection in ocr_results:

                text = detection[1]

                ocr_confidence = detection[2]


                cleaned = re.sub(
                    r"[^A-Z0-9]",
                    "",
                    text.upper()
                )


                print(
                    "OCR result:",
                    cleaned,
                    "| Confidence:",
                    round(ocr_confidence, 2)
                )


                if (
                    len(cleaned) >= 2
                    and ocr_confidence > best_confidence
                ):

                    plate_text = cleaned

                    best_confidence = ocr_confidence


            # -------------------------------------------------
            # MONGODB SEARCH
            # -------------------------------------------------

            if plate_text:

                ocr_detected += 1

                print(
                    "Recognized Plate:",
                    plate_text
                )


                vehicle = vehicles.find_one(
                    {
                        "plate_number": plate_text
                    }
                )


                if vehicle:

                    owner = vehicle["owner_name"]
                    phone = vehicle["phone"]
                    vehicle_type = vehicle["vehicle_type"]


                    print("Vehicle found!")
                    print("Owner:", owner)
                    print("Phone:", phone)
                    print("Vehicle Type:", vehicle_type)


                    label = (
                        plate_text
                        + " | "
                        + owner
                    )

                else:

                    print(
                        "Plate not found in MongoDB."
                    )


                    label = (
                        plate_text
                        + " | Not Registered"
                    )


            else:

                label = "OCR: No text"


            # -------------------------------------------------
            # DRAW BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                2
            )


            cv2.putText(
                frame,
                label,
                (
                    x1,
                    max(y1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )


    # -------------------------------------------------
    # SAVE FRAME
    # -------------------------------------------------

    output.write(frame)


# =====================================================
# 7. FINISH
# =====================================================

video.release()
output.release()
client.close()


print()
print("================================")
print("PROCESSING FINISHED")
print("================================")

print(
    "Total license plates detected:",
    plates_detected
)

print(
    "OCR readings obtained:",
    ocr_detected
)

print(
    "Output saved as:",
    "ocr_database_output.mp4"
)