import cv2
from ultralytics import YOLO
import os

print("Starting Traffic Violation Detector...")


# -------------------------------------------------
# LOAD YOLO MODEL
# -------------------------------------------------

model = YOLO("yolo11n.pt")

print("YOLO model loaded successfully.")


# -------------------------------------------------
# OPEN VIDEO
# -------------------------------------------------

video = cv2.VideoCapture("traffic.mp4")

if not video.isOpened():
    print("ERROR: Could not open traffic.mp4")
    exit()

print("Traffic video opened successfully.")


# -------------------------------------------------
# VIDEO SETTINGS
# -------------------------------------------------

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = video.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30


# -------------------------------------------------
# OUTPUT
# -------------------------------------------------

os.makedirs("output", exist_ok=True)

output_path = "output/vehicle_tracking.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# We process every 2nd frame, so adjust output FPS
output = cv2.VideoWriter(
    output_path,
    fourcc,
    fps / 2,
    (width, height)
)


# -------------------------------------------------
# VEHICLE CLASSES
# -------------------------------------------------

vehicle_classes = [2, 3, 5, 7]

# 2 = Car
# 3 = Motorcycle
# 5 = Bus
# 7 = Truck


# -------------------------------------------------
# PROCESSING SETTINGS
# -------------------------------------------------

frame_count = 0

PROCESS_EVERY_N_FRAMES = 2


print()
print("Vehicle detection and tracking started.")
print("GUI display disabled.")
print("Processing video...")


# -------------------------------------------------
# PROCESS VIDEO
# -------------------------------------------------

while True:

    ret, frame = video.read()

    if not ret:
        break

    frame_count += 1


    # -------------------------------------------------
    # PROCESS EVERY 2ND FRAME
    # -------------------------------------------------

    if frame_count % PROCESS_EVERY_N_FRAMES != 0:
        continue


    # -------------------------------------------------
    # YOLO TRACKING
    # -------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=vehicle_classes,
        conf=0.25,
        imgsz=640,
        verbose=False
    )


    # -------------------------------------------------
    # DRAW VEHICLES
    # -------------------------------------------------

    for result in results:

        if result.boxes is None:
            continue


        for box in result.boxes:

            # Coordinates
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Confidence
            confidence = float(
                box.conf[0]
            )


            if confidence < 0.25:
                continue


            # Class ID
            class_id = int(
                box.cls[0]
            )


            # Vehicle names
            names = {
                2: "Car",
                3: "Motorcycle",
                5: "Bus",
                7: "Truck"
            }


            vehicle_name = names.get(
                class_id,
                "Vehicle"
            )


            # -------------------------------------------------
            # TRACK ID
            # -------------------------------------------------

            if box.id is not None:

                track_id = int(
                    box.id[0]
                )

            else:

                track_id = 0


            # -------------------------------------------------
            # DRAW BOX
            # -------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # -------------------------------------------------
            # DRAW LABEL
            # -------------------------------------------------

            label = (
                f"{vehicle_name} "
                f"ID:{track_id} "
                f"{confidence:.2f}"
            )


            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 25)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # -------------------------------------------------
    # FRAME NUMBER
    # -------------------------------------------------

    cv2.putText(
        frame,
        f"Frame: {frame_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # -------------------------------------------------
    # SAVE FRAME
    # -------------------------------------------------

    output.write(frame)


    # -------------------------------------------------
    # PROGRESS
    # -------------------------------------------------

    if frame_count % 20 == 0:

        print(
            f"Processed frame: {frame_count}"
        )


# -------------------------------------------------
# CLEANUP
# -------------------------------------------------

video.release()
output.release()


print()
print("--------------------------------")
print("PROCESSING FINISHED")
print("--------------------------------")

print(
    "Processed frames:",
    frame_count
)

print(
    "Output saved to:",
    output_path
)
