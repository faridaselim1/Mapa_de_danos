import sqlite3
import requests
import time
import logging

# Database path
DB_PATH = "mapillary_data.db"

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# User-Agent (required by Nominatim to avoid being blocked)
HEADERS = {
    "User-Agent": "MyMapillaryApp/1.0 (farida.selim1@gmail.com)"  # ✅ Replace with your email
}

# Function to update missing addresses
def update_missing_addresses():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Fetch all records where location_type or full_address is missing
    c.execute("SELECT id, latitude, longitude FROM locations WHERE location_type = '' OR full_address = ''")
    rows = c.fetchall()

    if not rows:
        logging.info("No missing addresses to update.")
        return

    logging.info(f"Found {len(rows)} records to update...")

    for row in rows:
        image_id, latitude, longitude = row

        # Nominatim API request with headers
        nominatim_url = f"https://nominatim.openstreetmap.org/reverse?lat={latitude}&lon={longitude}&format=json"
        response = requests.get(nominatim_url, headers=HEADERS)

        if response.status_code == 200:
            location_data = response.json()
            location_type = location_data.get("type", "unknown")
            full_address = location_data.get("display_name", "Unknown Address")

            # ✅ Update database with fetched details
            c.execute(
                "UPDATE locations SET location_type = ?, full_address = ? WHERE id = ?",
                (location_type, full_address, image_id),
            )
            conn.commit()
            logging.info(f"Updated ID {image_id}: {location_type} - {full_address}")


    conn.close()
    logging.info("✅ Address update complete!")

# Run the script
if __name__ == "__main__":
    update_missing_addresses()

