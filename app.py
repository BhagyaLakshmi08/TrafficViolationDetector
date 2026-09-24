import cv2
from ultralytics import YOLO
import os

# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. OPEN VIDEO
# ============================================================

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("Error: Could not open traffic.mp4")
    exit()

print("Traffic video opened successfully.")


# ============================================================
# 3. VEHICLE CLASSES
# ============================================================

vehicle_classes = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}


# ============================================================
# 4. CREATE VIOLATION FOLDER
# ============================================================

os.makedirs("violations", exist_ok=True)


# ============================================================
# 5. TRACKING DATA
# ============================================================

previous_positions = {}

violated_vehicles = set()

violation_count = 0


# ============================================================
# 6. TRAFFIC LIGHT
# ============================================================

# For testing, assume the traffic light is RED.

traffic_light = "RED"


# ============================================================
# 7. START
# ============================================================

print("Vehicle tracking started.")
print("Traffic light:", traffic_light)
print("Red-light violation detection started.")
print("Press Q to quit.")


# ============================================================
# 8. PROCESS VIDEO
# ============================================================

while True:

    success, frame = video.read()

    if not success:
        print("Video finished.")
        break


    # ========================================================
    # FRAME SIZE
    # ========================================================

    height, width = frame.shape[:2]


    # ========================================================
    # STOP LINE
    # ========================================================

    # Position of the virtual stop line.
    # Change this ONLY if necessary after seeing the result.

    stop_line_y = int(height * 0.70)


    # Draw stop line

    cv2.line(
        frame,
        (0, stop_line_y),
        (width, stop_line_y),
        (0, 0, 255),
        4
    )


    cv2.putText(
        frame,
        "STOP LINE",
        (20, stop_line_y - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ========================================================
    # DISPLAY TRAFFIC LIGHT
    # ========================================================

    cv2.rectangle(
        frame,
        (width - 280, 20),
        (width - 20, 75),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        frame,
        "LIGHT: RED",
        (width - 250, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )


    # ========================================================
    # YOLO TRACKING
    # ========================================================

    results = model.track(
        frame,
        persist=True,
        classes=list(vehicle_classes.keys()),
        verbose=False
    )


    # ========================================================
    # PROCESS VEHICLES
    # ========================================================

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Tracking ID required

            if box.id is None:
                continue


            # ------------------------------------------------
            # VEHICLE INFORMATION
            # ------------------------------------------------

            class_id = int(box.cls[0])

            track_id = int(box.id[0])

            confidence = float(box.conf[0])


            if class_id not in vehicle_classes:
                continue


            vehicle_name = vehicle_classes[class_id]


            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # ------------------------------------------------
            # CENTER POINT
            # ------------------------------------------------

            center_x = (x1 + x2) // 2

            center_y = (y1 + y2) // 2


            # =================================================
            # CHECK CROSSING
            # =================================================

            crossed_line = False


            if track_id in previous_positions:

                previous_y = previous_positions[track_id]


                # ------------------------------------------------
                # CROSSING IN EITHER DIRECTION
                # ------------------------------------------------

                moved_down = (
                    previous_y < stop_line_y
                    and center_y >= stop_line_y
                )


                moved_up = (
                    previous_y > stop_line_y
                    and center_y <= stop_line_y
                )


                if moved_down or moved_up:

                    crossed_line = True


            # =================================================
            # RED LIGHT VIOLATION
            # =================================================

            if (
                traffic_light == "RED"
                and crossed_line
                and track_id not in violated_vehicles
            ):

                # Register violation

                violated_vehicles.add(track_id)

                violation_count += 1


                # ------------------------------------------------
                # TERMINAL MESSAGE
                # ------------------------------------------------

                print()
                print("========================================")
                print("     RED LIGHT VIOLATION DETECTED")
                print("========================================")
                print("Vehicle:", vehicle_name)
                print("Vehicle ID:", track_id)
                print("Confidence:", round(confidence, 2))
                print("Violation number:", violation_count)
                print("========================================")


                # ------------------------------------------------
                # SAVE EVIDENCE
                # ------------------------------------------------

                filename = (
                    f"violations/"
                    f"red_light_violation_ID_{track_id}.jpg"
                )


                cv2.imwrite(
                    filename,
                    frame
                )


                print(
                    "Evidence saved:",
                    filename
                )


            # =================================================
            # SAVE CURRENT POSITION
            # =================================================

            previous_positions[track_id] = center_y


            # =================================================
            # VEHICLE DISPLAY
            # =================================================

            if track_id in violated_vehicles:

                box_color = (0, 0, 255)

                label = (
                    f"VIOLATION | "
                    f"{vehicle_name} | "
                    f"ID:{track_id}"
                )

            else:

                box_color = (0, 255, 0)

                label = (
                    f"{vehicle_name} | "
                    f"ID:{track_id} | "
                    f"{confidence:.2f}"
                )


            # =================================================
            # DRAW VEHICLE BOX
            # =================================================

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                3
            )


            # =================================================
            # DRAW LABEL
            # =================================================

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


            # =================================================
            # DRAW CENTER POINT
            # =================================================

            cv2.circle(
                frame,
                (center_x, center_y),
                6,
                (255, 0, 0),
                -1
            )


    # ========================================================
    # VIOLATION COUNT
    # ========================================================

    cv2.rectangle(
        frame,
        (10, 10),
        (270, 60),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        frame,
        f"Violations: {violation_count}",
        (20, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # ========================================================
    # ALERT
    # ========================================================

    if violation_count > 0:

        cv2.putText(
            frame,
            "RED LIGHT VIOLATION DETECTED",
            (width // 2 - 330, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3
        )


    # ========================================================
    # DISPLAY VIDEO
    # ========================================================

    cv2.imshow(
        "Traffic Violation Detector",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

video.release()

cv2.destroyAllWindows()


print()
print("========================================")
print("Program stopped.")
print("Total violations:", violation_count)
print("========================================")
