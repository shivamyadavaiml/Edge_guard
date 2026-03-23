from ultralytics import YOLO
import cv2
import time
import csv
import os

# Load YOLO model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# Create folder for incident images
if not os.path.exists("incident_images"):
    os.makedirs("incident_images")

# Create / open log file
file_exists = os.path.isfile("incident_log.csv")
log_file = open("incident_log.csv", mode="a", newline="")
logger = csv.writer(log_file)

if not file_exists:
    logger.writerow(["Time", "Object", "Distance_Level", "Alert_Status", "Image_File"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4)
    annotated = frame.copy()

    alert_status = "SAFE"

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1

        # Distance logic
        if width > 300:
            distance_level = "VERY CLOSE"
            alert_status = "DANGER"
        elif width > 150:
            distance_level = "CLOSE"
            alert_status = "WARNING"
        else:
            distance_level = "FAR"

        # Draw box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(annotated,
                    f"{label} | {distance_level}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2)

        # If DANGER detected → Save image + log
        if alert_status == "DANGER":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_name = f"incident_{timestamp}.jpg"
            image_path = os.path.join("incident_images", image_name)

            cv2.imwrite(image_path, annotated)

            logger.writerow([
                time.strftime("%Y-%m-%d %H:%M:%S"),
                label,
                distance_level,
                alert_status,
                image_name
            ])

    # Show alert on screen
    cv2.putText(annotated,
                f"ALERT: {alert_status}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                3)

    cv2.imshow("Edge Guard AI", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
log_file.close()
cv2.destroyAllWindows()
