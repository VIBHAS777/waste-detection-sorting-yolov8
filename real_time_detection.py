

import cv2
from ultralytics import YOLO

model = YOLO("runs/detect/biodegradable_detection/weights/yolov8n.pt")

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            # Confidence score
            conf = float(box.conf[0])
            # Class ID
            cls = int(box.cls[0])
            # ✅ Convert tensor to int to avoid KeyError
            class_name = model.names[int(cls)]

            # Define colors for classes
            color = (0, 255, 0) if class_name.lower() == 'biodegradable' else (0, 0, 255)

            # Draw bounding box
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            # Add label text
            cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ✅ Display frame
    cv2.imshow('YOLOv8 Real-Time Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
cv2.destroyAllWindows()
