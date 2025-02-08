import os
import sqlite3
import requests
import logging
import time

# ✅ Configuration
DB_PATH = "mapillary_data.db"
EXPORTED_IMAGES_DIR = "exported_images"
MAPILLARY_ACCESS_TOKEN = "MLY|9058976994140764|e1f2b718b58e5c12336be21a6e3459fb"

# ✅ Ensure `exported_images/` exists
os.makedirs(EXPORTED_IMAGES_DIR, exist_ok=True)

# ✅ Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def fetch_missing_images():
    """ Fetch all missing images and store them in the database. """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ✅ Select IDs of images where image_1024 is NULL
    c.execute("SELECT id FROM locations WHERE image_1024 IS NULL")
    missing_images = [row[0] for row in c.fetchall()]
    conn.close()

    if not missing_images:
        logging.info("✅ No missing images to download.")
        return

    logging.info(f"📥 Found {len(missing_images)} missing images to download.")

    for image_id in missing_images:
        try:
            # ✅ Fetch image URL from Mapillary API
            fields = ["id", "thumb_1024_url"]
            detail_url = f"https://graph.mapillary.com/{image_id}?access_token={MAPILLARY_ACCESS_TOKEN}&fields={','.join(fields)}"
            response = requests.get(detail_url)

            if response.status_code == 200:
                json_data = response.json()
                thumb_1024_url = json_data.get("thumb_1024_url")

                if thumb_1024_url:
                    logging.info(f"📥 Downloading image for {image_id}...")

                    # ✅ Download the image
                    img_response = requests.get(thumb_1024_url)
                    if img_response.status_code == 200:
                        image_data = img_response.content

                        # ✅ Save to `exported_images/`
                        image_path = os.path.join(EXPORTED_IMAGES_DIR, f"{image_id}.png")
                        with open(image_path, "wb") as img_file:
                            img_file.write(image_data)

                        logging.info(f"✅ Image {image_id} saved to {image_path}")

                        # ✅ Store image in database
                        save_image_to_db(image_id, image_data)

                    else:
                        logging.error(f"❌ Failed to download image for {image_id}")

                else:
                    logging.error(f"⚠️ No image URL found for {image_id}")

            else:
                logging.error(f"❌ Failed to fetch image details for {image_id}. Status: {response.status_code}")

            time.sleep(1)  # ✅ Add delay to prevent API rate limits

        except Exception as e:
            logging.error(f"❌ Error processing image {image_id}: {e}")


def save_image_to_db(image_id, image_data):
    """ Save image data as BLOB in SQLite database. """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("UPDATE locations SET image_1024 = ? WHERE id = ?", (sqlite3.Binary(image_data), image_id))
        conn.commit()
        conn.close()
        logging.info(f"📌 Image {image_id} stored in the database.")
    except Exception as e:
        logging.error(f"❌ Error storing image {image_id} in database: {e}")


if __name__ == "__main__":
    logging.info("🚀 Running image download script...")
    fetch_missing_images()
    logging.info("✅ Image download complete.")
