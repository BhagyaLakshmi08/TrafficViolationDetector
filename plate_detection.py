import cv2
from ultralytics import YOLO

# Load license plate model
model = YOLO("license-plate-finetune-v1n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("Error: Could not open traffic.mp4")
    exit()

print("Traffic video opened successfully.")
print("License plate detection started.")
print("Press Q to quit.")

while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    # Detect license plates
    results = model(
        frame,
        verbose=False
    )

    # Process detections
    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(box.conf[0])

            # Ignore weak detections
            if confidence < 0.40:
                continue

            # Plate bounding box
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            # Draw plate box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 255),
                3
            )

            # Label
            label = f"LICENSE PLATE {confidence:.2f}"

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

    # Show video
    cv2.imshow(
        "License Plate Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# Cleanup
video.release()
cv2.destroyAllWindows()

print("Program stopped.")