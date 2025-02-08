import os
import cv2
import sqlite3
import numpy as np
from ultralytics import YOLO

# Database Path
DB_PATH = "mapillary_data.db"

# Load YOLO models
door_detection_model = YOLO(r"D:/5.Semester/Freie_Entwurf/3.Kolloq/mapillary/runs/detect/train2/weights/best.pt")  # Detect doors
damage_classification_model = YOLO(r"D:/5.Semester/Freie_Entwurf/3.Kolloq/mapillary/runs/detect/train4/weights/best.pt")  # Classify damage severity

# Connect to SQLite
conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# Fetch images from the database
c.execute("SELECT id, image FROM locations WHERE detected IS NULL")  # Process only new images
rows = c.fetchall()

print(f"📸 Found {len(rows)} new images to process.")

for row in rows:
    image_id, image_data = row

    # Convert BLOB to image
    nparr = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print(f"\n🔍 Processing Image ID: {image_id}")

    # Step 1: Run door detection model
    results = door_detection_model.predict(source=image, save=False)

    detected_doors = 0
    cropped_images = []

    # Process each detected object
    for box in results[0].boxes:
        bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = map(int, box.xyxy[0])
        predicted_class_idx = int(box.cls[0])
        predicted_class_name = door_detection_model.names[predicted_class_idx]

        if "door" in predicted_class_name:
            detected_doors += 1
            print(f"✅ Detected {predicted_class_name} at ({bbox_xmin}, {bbox_ymin})")

            # Crop detected door
            cropped_door = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

            # Convert cropped image to BLOB format
            _, encoded_image = cv2.imencode('.jpg', cropped_door)
            cropped_images.append(encoded_image.tobytes())

            # Step 2: Run damage classification model
            classification_results = damage_classification_model.predict(source=cropped_door, save=False)
            damage_label_idx = int(classification_results[0].boxes.cls[0])
            damage_label = damage_classification_model.names[damage_label_idx]

            print(f"📌 Damage Severity: {damage_label}")

            # Save to database
            c.execute("""
                UPDATE locations 
                SET detected = ?, cropped = ?, damage_severity = ? 
                WHERE id = ?
            """, ("door", encoded_image.tobytes(), damage_label, image_id))
            conn.commit()

    if detected_doors == 0:
        print(f"❌ No doors detected in Image ID {image_id}. Marking as 'unidentified'.")
        c.execute("UPDATE locations SET detected = 'unidentified' WHERE id = ?", (image_id,))
        conn.commit()

conn.close()
print("✅ Processing complete! Database updated.")