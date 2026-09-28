import cv2
from ultralytics import YOLO


print("Loading license plate detector...")

model = YOLO("license_plate_detector.pt")

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()


frame_number = 0
saved = False


while frame_number < 100:

    ret, frame = video.read()

    if not ret:
        break

    frame_number += 1

    results = model(
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

            # Make the crop larger
            plate = cv2.resize(
                plate,
                None,
                fx=8,
                fy=8,
                interpolation=cv2.INTER_CUBIC
            )

            filename = "detected_plate.jpg"

            cv2.imwrite(
                filename,
                plate
            )

            print()
            print("Plate detected!")
            print("Frame:", frame_number)
            print("Confidence:", confidence)
            print("Saved:", filename)

            saved = True
            break

        if saved:
            break

    if saved:
        break


video.release()


if not saved:
    print("No license plate detected in first 100 frames.")
else:
    print("Plate crop saved successfully.")