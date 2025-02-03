from flask import Flask, request, jsonify, render_template, send_file
import requests
from flask import send_from_directory
import os
import sqlite3
import json
from PIL import Image
import io
import csv
import logging
import subprocess 
from werkzeug.utils import secure_filename
import os
import sqlite3
import cv2
import numpy as np
from ultralytics import YOLO


# Set up logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure the upload folder exists
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

MAPILLARY_ACCESS_TOKEN = "MLY|9058976994140764|e1f2b718b58e5c12336be21a6e3459fb"
DB_PATH = "mapillary_data.db"

# Initialize database
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS locations (
                 id TEXT PRIMARY KEY,
                 latitude REAL,
                 longitude REAL,
                 captured_at TEXT,
                 sequence_id TEXT,
                 json_data TEXT,
                 image_1024 BLOB,
                 location_type TEXT DEFAULT 'unknown',
                 full_address TEXT DEFAULT 'Unknown Address')''')
    conn.commit()
    conn.close()


import time

def get_address(lat, lon):
    """Fetch the address type and full address from Nominatim API"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {
        "User-Agent": "MapillaryImageDownloader/1.0 (contact: farida.selim1@gmail.com)",
        "Referer": "https://github.com/faridaselim1/reurb"
    }

    delay = 1  # Start with a 1-second delay
    for attempt in range(5):  # Retry up to 5 times
        try:
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                address_type = data.get("type", "unknown")
                full_address = data.get("display_name", "Unknown Address")
                return address_type, full_address

            elif response.status_code == 403:  # Blocked due to rate limit
                logging.warning(f"🚨 Nominatim API blocked request! Retrying in {delay}s...")
                time.sleep(delay)  # Exponential backoff
                delay *= 2  # Increase delay (1s → 2s → 4s → ...)
            else:
                logging.error(f"⚠️ Failed to fetch address: {response.status_code} - {response.text}")
                break  # Stop retrying if it's another error (like 500)

        except Exception as e:
            logging.error(f"❌ Error fetching address: {e}")

    return "unknown", "Unknown Address"  # Return defaults if all attempts fail



# Store location data in database
import time  # ✅ Import time for sleep

def store_location(image_data):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    try:
        logging.info(f"📝 Processing image: {json.dumps(image_data, indent=2)}")  # Log full image data

        if isinstance(image_data, str):
            try:
                image_data = json.loads(image_data)
            except json.JSONDecodeError:
                logging.error(f"❌ Error decoding JSON: {image_data}")
                return
        
        if not isinstance(image_data, dict):
            logging.error(f"❌ Invalid image_data format (Not a dict): {image_data}")
            return

        if "id" not in image_data:
            logging.error(f"❌ Image data missing 'id': {image_data}")
            return

        image_id = image_data["id"]
        coordinates = image_data.get("geometry", {}).get("coordinates", [None, None])

        if len(coordinates) < 2 or None in coordinates:
            logging.error(f"❌ Invalid geometry for image {image_id}: {coordinates}")
            return
        
        latitude, longitude = coordinates[1], coordinates[0]

        # ✅ Ensure Image is Downloaded and Saved
        thumb_url = image_data.get("thumb_1024_url", None)
        image_1024 = None

        if thumb_url:
            logging.info(f"📥 Downloading image from: {thumb_url}")
            response = requests.get(thumb_url)
            if response.status_code == 200:
                image_1024 = response.content

                # ✅ Save Image to exported_images folder
                os.makedirs("exported_images", exist_ok=True)
                image_path = os.path.join("exported_images", f"{image_id}.png")
                with open(image_path, "wb") as img_file:
                    img_file.write(image_1024)
                logging.info(f"✅ Saved image {image_id} to {image_path}")
            else:
                logging.error(f"❌ Error downloading image {image_id}: {response.status_code}")

        # ✅ Insert into Database
        c.execute('''INSERT OR REPLACE INTO locations 
                     (id, latitude, longitude, json_data, image_1024) 
                     VALUES (?, ?, ?, ?, ?)''',
                  (image_id, latitude, longitude, json.dumps(image_data), image_1024))
        conn.commit()
        logging.info(f"✅ Stored image ID: {image_id} at ({latitude}, {longitude})")

    except Exception as e:
        logging.error(f"❌ Error processing image {image_data.get('id', 'Unknown')}: {e}")

    finally:
        conn.close()



def get_address(lat, lon):
    """Fetch the address type and full address from Nominatim API"""
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json"
    headers = {
        "User-Agent": "MapillaryImageDownloader/1.0 (contact: farida.selim1@gmail.com)"
    }

    for attempt in range(3):  # ✅ Retry up to 3 times
        try:
            time.sleep(1)  # ✅ Add delay to prevent rate limiting
            response = requests.get(url, headers=headers)

            if response.status_code == 200:
                data = response.json()
                address_type = data.get("type", "unknown")
                full_address = data.get("display_name", "Unknown Address")
                return address_type, full_address

            elif response.status_code == 403:
                logging.error("🚨 Nominatim API blocked your request! Retrying...")
                time.sleep(3)  # ✅ Wait before retrying
            else:
                logging.error(f"⚠️ Failed to fetch address: {response.status_code} - {response.text}")

        except Exception as e:
            logging.error(f"❌ Error fetching address: {e}")

    return "unknown", "Unknown Address"  # Return default if all attempts fail


def update_missing_addresses():
    """Find locations without an address and update them using Nominatim"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id, latitude, longitude FROM locations WHERE full_address IS NULL OR full_address = 'Unknown Address'")
    rows = c.fetchall()

    if not rows:
        logging.info("✅ No missing addresses to update.")
        return

    logging.info(f"Found {len(rows)} locations missing an address. Fetching now...")

    for image_id, lat, lon in rows:
        address_type, full_address = get_address(lat, lon)

        c.execute("UPDATE locations SET location_type = ?, full_address = ? WHERE id = ?",
                  (address_type, full_address, image_id))
        conn.commit()
        logging.info(f"📌 Updated {image_id} with address: {full_address}")

    conn.close()
    logging.info("✅ All missing addresses have been updated!")



@app.route("/")
def home():
    return render_template("home.html")  # Serve homepage

@app.route("/map")
def map_page():
    return render_template("map.html")  # Serve map interface


@app.route("/add-pin", methods=["POST"])
def add_pin():
    data = request.form
    file = request.files.get('image')

    if not data:
        return jsonify({"error": "Missing data"}), 400

    try:
        latitude = float(data.get('latitude'))
        longitude = float(data.get('longitude'))
        pin_id = f"custom_{latitude}_{longitude}"  # Unique ID for the pin

        image_blob = None
        image_filename = f"{pin_id}.png"
        image_path = os.path.join("exported_images", image_filename)  # ✅ Save here

        if file:
            # ✅ Save image to exported_images folder
            file.save(image_path)

            # ✅ Convert to binary for DB storage
            with open(image_path, "rb") as img_file:
                image_blob = img_file.read()

        # ✅ Insert into DB with `detected = NULL` so AI processes it
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''INSERT OR REPLACE INTO locations 
                     (id, latitude, longitude, image_1024, detected, damage_severity)
                     VALUES (?, ?, ?, ?, NULL, "no_damage")''',
                  (pin_id, latitude, longitude, image_blob))
        conn.commit()
        conn.close()

        print(f"✅ Manually added image {pin_id} saved to exported_images and marked for AI processing.")

        # ✅ Ensure AI Processing picks up the new image
        process_images_from_db()

        return jsonify({"message": "Pin added successfully and AI processing started!", "pin_id": pin_id}), 200
    except Exception as e:
        logging.error(f"Error adding pin: {e}")
        return jsonify({"error": "Failed to add pin"}), 500


@app.route("/save-area", methods=["POST"])
def save_area():
    data = request.json
    if not data or "bounds" not in data:
        return jsonify({"error": "Invalid data"}), 400

    bounds = data["bounds"]
    south, west, north, east = bounds

    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO areas (south, west, north, east) VALUES (?, ?, ?, ?)", 
                  (south, west, north, east))
        conn.commit()
        conn.close()

        return jsonify({"message": "Area saved successfully!"}), 200
    except Exception as e:
        logging.error(f"Error saving area: {e}")
        return jsonify({"error": "Failed to save area"}), 500
    

@app.route("/download-images", methods=["POST"])
def download_images():
    data = request.json
    if not data or "bounds" not in data:
        return jsonify({"error": "Invalid data"}), 400

    bounds = data["bounds"]
    south, west, north, east = bounds

    logging.info(f"📦 Fetching images for bounds: South={south}, West={west}, North={north}, East={east}")

    total_processed = 0
    per_page = 100
    has_more = True
    after = None

    while has_more:
        url = f"https://graph.mapillary.com/images?access_token={MAPILLARY_ACCESS_TOKEN}&bbox={west},{south},{east},{north}&limit={per_page}"
        if after:
            url += f"&after={after}"

        response = requests.get(url)
        logging.info(f"🔍 API URL: {url}")

        if response.status_code != 200:
            logging.error(f"❌ Error fetching images: {response.content}")
            break

        response_data = response.json()
        images = response_data.get("data", [])

        logging.info(f"📸 Number of images fetched in this batch: {len(images)}")

        if not images:
            break

        for image in images:
            logging.info(f"📝 Raw image data received: {image}")

            if isinstance(image, dict):
                try:
                    store_location(image)  # ✅ Ensure images are stored
                    total_processed += 1
                except Exception as e:
                    logging.error(f"❌ Failed to store image: {e}")
            else:
                logging.error(f"⚠️ Skipping invalid image format: {image}")

        after = response_data.get("paging", {}).get("cursors", {}).get("after")
        has_more = bool(after)

    logging.info(f"✅ Total processed images: {total_processed}")

    # ✅ STEP 2: Run `download_missing.py`
    try:
        logging.info("🚀 Running 'download_missing.py' to fetch missing images...")
        subprocess.run(["python", "download_missing.py"], check=True)
        logging.info("✅ Finished running 'download_missing.py'")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error running 'download_missing.py': {e}")
        return jsonify({"error": "Failed to download missing images"}), 500

    # ✅ STEP 3: Run AI Model for Classification
    try:
        logging.info("🤖 Running AI Model to classify doors and detect damage...")
        process_images_from_db()  # Your existing AI function
        logging.info("✅ AI Model processing complete!")
    except Exception as e:
        logging.error(f"❌ Error running AI Model: {e}")
        return jsonify({"error": "AI Model processing failed"}), 500

    # ✅ STEP 4: Run `update_addresses.py`
    try:
        logging.info("📍 Running 'update_addresses.py' to fetch location addresses...")
        subprocess.run(["python", "update_addresses.py"], check=True)
        logging.info("✅ Finished running 'update_addresses.py'")
    except subprocess.CalledProcessError as e:
        logging.error(f"❌ Error running 'update_addresses.py': {e}")
        return jsonify({"error": "Failed to update addresses"}), 500

    return jsonify({"message": f"Successfully processed {total_processed} images"}), 200


@app.route("/download-missing-images", methods=["POST"])
def download_missing_images():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # ✅ Select records that are missing images (image_1024 is NULL)
    c.execute("SELECT id FROM locations WHERE image_1024 IS NULL")
    rows = c.fetchall()
    conn.close()

    if not rows:
        return jsonify({"message": "No missing images to download."}), 200

    missing_image_ids = [row[0] for row in rows]
    logging.info(f"Found {len(missing_image_ids)} missing images to download.")

    total_downloaded = 0

    for image_id in missing_image_ids:
        try:
            # ✅ Fetch image URL from Mapillary
            fields = ['id', 'thumb_1024_url']
            detail_url = f"https://graph.mapillary.com/{image_id}?access_token={MAPILLARY_ACCESS_TOKEN}&fields={','.join(fields)}"
            detail_response = requests.get(detail_url)

            if detail_response.status_code == 200:
                json_data = detail_response.json()
                thumb_1024_url = json_data.get("thumb_1024_url")

                if thumb_1024_url:
                    logging.info(f"Downloading image for {image_id}: {thumb_1024_url}")  
                    thumb_response = requests.get(thumb_1024_url)

                    if thumb_response.status_code == 200:
                        image_1024 = thumb_response.content

                        # ✅ Save image to exported_images folder
                        os.makedirs("exported_images", exist_ok=True)  # Ensure folder exists
                        image_path = os.path.join("exported_images", f"{image_id}.png")

                        with open(image_path, "wb") as img_file:
                            img_file.write(image_1024)

                        logging.info(f"✅ Saved image {image_id} to {image_path}")

                        # ✅ Store image in database as BLOB
                        conn = sqlite3.connect(DB_PATH)
                        c = conn.cursor()
                        c.execute("UPDATE locations SET image_1024 = ? WHERE id = ?", (sqlite3.Binary(image_1024), image_id))
                        conn.commit()
                        conn.close()

                        total_downloaded += 1
                    else:
                        logging.error(f"❌ Failed to download image for {image_id}")
                else:
                    logging.error(f"⚠️ No image URL found for {image_id}")

            else:
                logging.error(f"⚠️ Failed to fetch image details for {image_id}. Status: {detail_response.status_code}")

        except Exception as e:
            logging.error(f"❌ Error processing image {image_id}: {e}")

    return jsonify({"message": f"Successfully downloaded {total_downloaded} images"}), 200



@app.route("/show-pins", methods=["GET"])
def show_pins():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, latitude, longitude, damage_severity, location_type, full_address, image_1024 FROM locations")
    rows = c.fetchall()
    conn.close()

    pins = [
        {
            "id": row[0],
            "latitude": row[1],
            "longitude": row[2],
            "damage_severity": row[3] if row[3] else "no damage",
            "location_type": row[4] if row[4] else "unknown",
            "full_address": row[5] if row[5] else "Unknown Address",
            "image_url": f"/pin-image/{row[0]}" if row[6] else None  # ✅ Only include if image exists
        }
        for row in rows
    ]

    return jsonify(pins)



@app.route("/pin-image/<string:pin_id>", methods=["GET"])
def pin_image(pin_id):
    """ Serve the image from exported_images or database """
    image_path = os.path.join("exported_images", f"{pin_id}.png")

    # ✅ If the image exists in the folder, serve it
    if os.path.exists(image_path):
        return send_file(image_path, mimetype="image/png", as_attachment=False)

    # ✅ If not found in exported_images, try fetching from DB
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT image_1024 FROM locations WHERE id = ?", (pin_id,))
    row = c.fetchone()
    conn.close()

    if row and row[0]:
        return send_file(io.BytesIO(row[0]), mimetype="image/png", as_attachment=False)

    return jsonify({"error": f"Image not found for ID: {pin_id}"}), 404



@app.route("/export-to-csv", methods=["GET"])
def export_to_csv():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    rows = c.execute("SELECT * FROM locations").fetchall()
    column_names = [description[0] for description in c.description]

    csv_path = "locations.csv"
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(column_names)
        writer.writerows(rows)

    conn.close()
    return jsonify({"message": f"Data exported to {csv_path}"})

@app.route("/update-damage-severity", methods=["POST"])
def update_damage_severity():
    data = request.json
    if not data or "id" not in data or "damage_severity" not in data:
        return jsonify({"error": "Invalid data"}), 400

    image_id = data["id"]
    damage_severity = data["damage_severity"]

    if damage_severity not in ["severe", "minor", "no damage"]:
        return jsonify({"error": "Invalid damage severity value"}), 400

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute(
            "UPDATE locations SET damage_severity = ? WHERE id = ?",
            (damage_severity, image_id),
        )
        conn.commit()
        return jsonify({"message": "Damage severity updated successfully"}), 200
    except Exception as e:
        logging.error(f"Error updating damage severity: {e}")
        return jsonify({"error": "Failed to update damage severity"}), 500
    finally:
        conn.close()


# Define the function
def add_damage_severity_column():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        # Add the new column if it doesn't exist
        c.execute("ALTER TABLE locations ADD COLUMN damage_severity TEXT DEFAULT 'no damage'")
        conn.commit()
        logging.info("Added 'damage_severity' column to the locations table.")
    except sqlite3.OperationalError as e:
        logging.warning(f"'damage_severity' column might already exist. {e}")
    finally:
        conn.close()

# Route to serve color marker icons
@app.route('/color_markers/<path:filename>')
def serve_color_markers(filename):
    return send_from_directory('color_markers', filename)  # ✅ Fix: Correct function return

def add_address_columns():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE locations ADD COLUMN location_type TEXT DEFAULT 'unknown'")
        c.execute("ALTER TABLE locations ADD COLUMN full_address TEXT DEFAULT 'Unknown Address'")
        conn.commit()
        logging.info("Added 'location_type' and 'full_address' columns to the locations table.")
    except sqlite3.OperationalError as e:
        logging.warning(f"Columns might already exist: {e}")
    finally:
        conn.close()

# ✅ Run this once when the app starts
add_address_columns()


# Initialize the database and add the column
init_db()




# Paths
DB_PATH = "mapillary_data.db"
EXPORTED_IMAGES_DIR = "exported_images"

# Load YOLO models
DETECTION_MODEL_PATH = "D:/5.Semester/Freie_Entwurf/3.Kolloq/mapillary/runs/detect/train2/weights/best.pt"  # Detect doors
CLASSIFICATION_MODEL_PATH = "D:/5.Semester/Freie_Entwurf/3.Kolloq/mapillary/runs/detect/train4/weights/best.pt"  # Detect damage severity

detector = YOLO(DETECTION_MODEL_PATH)  # Model for detecting doors
classifier = YOLO(CLASSIFICATION_MODEL_PATH)  # Model for detecting damage severity


def process_images_from_db():
    """ Process new images, detect doors, crop them, detect damage severity, and update the database """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ Get images that have NOT been processed (detected is NULL)
    c.execute("SELECT id FROM locations WHERE detected IS NULL")
    unprocessed_images = {row[0] for row in c.fetchall()}

    if not unprocessed_images:
        print("✅ No new images to process.")
        return

    for image_id in unprocessed_images:
        image_path = os.path.join("exported_images", f"{image_id}.png")

        # ✅ FIX: If the image is missing, restore it from the database
        if not os.path.exists(image_path):
            print(f"⚠️ Image {image_path} not found, restoring from DB...")
            c.execute("SELECT image_1024 FROM locations WHERE id = ?", (image_id,))
            row = c.fetchone()

            if row and row[0]:
                with open(image_path, "wb") as img_file:
                    img_file.write(row[0])  # ✅ Restore the image from DB
                print(f"✅ Restored missing image {image_id} to {image_path}")

        # ✅ Re-check if the file exists after attempting restore
        if not os.path.exists(image_path):
            print(f"❌ Image {image_id} still missing, skipping...")
            continue

        print(f"\n🔍 Processing new image {image_id}...")

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Error reading {image_id}. Skipping.")
            continue

        # Run detection model (detect doors)
        results = detector.predict(source=image_path, save=False)

        detected_doors = 0
        cropped_images = []  # Store cropped door images

        # Process each detected object
        for box in results[0].boxes:
            bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = map(int, box.xyxy[0])
            predicted_class_idx = int(box.cls[0])
            predicted_class_name = detector.names[predicted_class_idx]

            if "door" in predicted_class_name:
                detected_doors += 1
                print(f"✅ Detected {predicted_class_name} at ({bbox_xmin}, {bbox_ymin})")

                # Crop the detected door
                cropped_door = image[bbox_ymin:bbox_ymax, bbox_xmin:bbox_xmax]

                # Convert cropped image to bytes for database storage
                _, cropped_encoded = cv2.imencode('.jpg', cropped_door)
                cropped_blob = cropped_encoded.tobytes()
                cropped_images.append(cropped_blob)

        # ✅ Update database with detection result
        detected_type = "door" if detected_doors > 0 else "unidentified"
        c.execute("UPDATE locations SET detected = ? WHERE id = ?", (detected_type, image_id))
        print(f"📌 Updated detected column: {detected_type}")

        # ✅ Store cropped images in the DB
        if cropped_images:
            c.execute("UPDATE locations SET cropped = ? WHERE id = ?", (cropped_images[0], image_id))
            print("📁 Cropped door stored in DB.")

            # Convert cropped image to array
            cropped_img_array = np.frombuffer(cropped_images[0], np.uint8)
            cropped_img = cv2.imdecode(cropped_img_array, cv2.IMREAD_COLOR)

            # Run Damage Detection Model on Cropped Doors
            damage_results = classifier.predict(source=cropped_img, save=False)

            # Check if any damage classes were detected
            damage_severity = "unknown"
            if damage_results and len(damage_results[0].boxes) > 0:
                highest_confidence_box = max(damage_results[0].boxes, key=lambda b: b.conf[0])
                damage_class_idx = int(highest_confidence_box.cls.item())
                damage_severity = classifier.names[damage_class_idx]

            print(f"📌 Damage Severity Detected: {damage_severity}")

            # ✅ Update the database with the detected damage severity
            c.execute("UPDATE locations SET damage_severity = ? WHERE id = ?", (damage_severity, image_id))

        conn.commit()

    conn.close()
    print("✅ All new images processed. Existing images were skipped.")



# Run AI processing function
process_images_from_db()

@app.route("/get-locations", methods=["GET"])
def get_locations():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, latitude, longitude, detected, damage_severity, location_type, full_address FROM locations")
    rows = c.fetchall()
    conn.close()

    features = []
    for row in rows:
        image_id, lat, lon, detected, damage_severity, location_type, full_address = row

        # ✅ Make sure image URL is included
        image_url = f"/pin-image/{image_id}" if detected != "unidentified" else None

        # ✅ Default classification for unidentified images
        if detected == "unidentified":
            damage_severity = "no_damage"

        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat]
            },
            "properties": {
                "id": image_id,
                "latitude": lat,
                "longitude": lon,
                "detected": detected,
                "damage_severity": damage_severity,
                "location_type": location_type,
                "full_address": full_address,
                "image_url": image_url  # ✅ Image should be included
            }
        })

    return jsonify({"type": "FeatureCollection", "features": features})


if __name__ == "__main__":
    print("\n🚀 Server running! Open in your browser: http://127.0.0.1:5000/")

    # ✅ Process new images only (skip old ones)
    process_images_from_db()

    app.run(debug=True, host="0.0.0.0", port=5000)
