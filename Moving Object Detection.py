
import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox

# Initialize video capture
cap = cv2.VideoCapture("F:\Traffic.mp4")  # Replace with the path to your video file

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects in the frame
    bbox, labels, conf = cv.detect_common_objects(frame, confidence=0.5, model='yolov3')

    # Draw bounding boxes and labels on the frame
    out = draw_bbox(frame, bbox, labels, conf, write_conf=True)

    # Display the resulting frame
    cv2.imshow("Object Detection", out)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()






