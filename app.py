
import cv2

# Open the traffic video
video = cv2.VideoCapture("traffic.mp4")

# Check whether the video opened successfully
if not video.isOpened():
    print("Error: Could not open the video.")
    exit()

print("Traffic video opened successfully!")
print("Press Q to quit.")

while True:

    # Read one frame from the video
    success, frame = video.read()

    # Stop when the video ends
    if not success:
        print("Video finished.")
        break

    # Display the current frame
    cv2.imshow("Traffic Violation Detector", frame)

    # Press Q to stop the video
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
