import cv2
from ultralytics import YOLO
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Open traffic video
video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("Error: Could not open the video.")
    exit()

# Vehicle classes
vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# Folder to save violation evidence
os.makedirs("violations", exist_ok=True)

# Store previous Y position of each vehicle
previous_positions = {}

# Store vehicles that have already been detected as violations
violated_vehicles = set()

print("Vehicle tracking started.")
print("Traffic violation detection started.")
print("Press Q to quit.")

while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # ---------------------------------------
    # VIRTUAL VIOLATION LINE
    # ---------------------------------------

    violation_line_y = height // 2

    cv2.line(
        frame,
        (0, violation_line_y),
        (width, violation_line_y),
        (0, 0, 255),
        3
    )

    cv2.putText(
        frame,
        "VIOLATION LINE",
        (20, violation_line_y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )

    # ---------------------------------------
    # YOLO VEHICLE TRACKING
    # ---------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            if box.id is None:
                continue

            class_id = int(box.cls[0])
            track_id = int(box.id[0])
            confidence = float(box.conf[0])

            if class_id not in vehicle_classes:
                continue

            # Bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Find center of vehicle
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            vehicle_name = vehicle_classes[class_id]

            # ---------------------------------------
            # CHECK LINE CROSSING
            # ---------------------------------------

            if track_id in previous_positions:

                previous_y = previous_positions[track_id]

                # Vehicle moved from above line to below line
                crossed_line = (
                    previous_y < violation_line_y
                    and center_y >= violation_line_y
                )

                if crossed_line and track_id not in violated_vehicles:

                    print(
                        f"VIOLATION DETECTED - "
                        f"{vehicle_name} ID:{track_id}"
                    )

                    violated_vehicles.add(track_id)

                    # Draw violation box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 0, 255),
                        4
                    )

                    # Save evidence image
                    filename = f"violations/violation_ID_{track_id}.jpg"

                    cv2.imwrite(filename, frame)

                    print(f"Evidence saved: {filename}")

            # Save current position
            previous_positions[track_id] = center_y

            # ---------------------------------------
            # DISPLAY VEHICLE
            # ---------------------------------------

            if track_id in violated_vehicles:

                label = (
                    f"VIOLATION | "
                    f"{vehicle_name} "
                    f"ID:{track_id}"
                )

                box_color = (0, 0, 255)

            else:

                label = (
                    f"{vehicle_name} "
                    f"ID:{track_id} "
                    f"{confidence:.2f}"
                )

                box_color = (0, 255, 0)

            # Bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )

            # Label
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # Center point
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (255, 0, 0),
                -1
            )

    # ---------------------------------------
    # DISPLAY FRAME
    # ---------------------------------------

    cv2.imshow(
        "Traffic Violation Detector",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------
# CLEANUP
# ---------------------------------------

video.release()
cv2.destroyAllWindows()

print("Program stopped.")
