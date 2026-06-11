import time
import socket
import zipfile
import io
import os
import requests
import pandas as pd


# ======================================
# USER SETTINGS
# ======================================

API_KEY = "6ipk89gzKNfhShcI9c0j3L6gGA4KBYFx1h7Bexjy"
EMAIL = "hossein.omidi74@gmail.com"

# New York City coordinates
LAT = 40.7128
LON = -74.0060

# WKT uses POINT(longitude latitude), not lat/lon
WKT = f"POINT({LON} {LAT})"

# GOES Conus PSM v4 supports 5-minute data for 2018-2024
YEARS = [
    "2018",
    "2019",
    "2020",
    "2021",
    "2022",
    "2023",
    "2024",
]

INTERVAL = "5"

# Required for pvlib power: ghi, dhi, dni (+ air_temperature, wind_speed as
# exogenous states). cloud_type / clearsky_* / fill_flag are NSRDB's own
# physical weather labels used for diagnostics and plots (no synthetic data).
ATTRIBUTES = (
    "ghi,dhi,dni,"
    "air_temperature,"
    "wind_speed,"
    "relative_humidity,"
    "solar_zenith_angle,"
    "surface_albedo,"
    "surface_pressure,"
    "cloud_type,"
    "clearsky_ghi,clearsky_dhi,clearsky_dni,"
    "fill_flag"
)

DOMAIN = "developer.nlr.gov"

BASE_URL = (
    "https://developer.nlr.gov/api/nsrdb/v2/solar/"
    "nsrdb-GOES-conus-v4-0-0-download.json"
)

OUTPUT_FOLDER = "nsrdb_newyork_5min"


# ======================================
# HELPER FUNCTIONS
# ======================================

def log(message):
    print(message, flush=True)


def test_dns():
    log("\n1) Testing DNS...")
    try:
        ip = socket.gethostbyname(DOMAIN)
        log(f"OK: DNS works. IP = {ip}")
        return True
    except Exception as e:
        log("FAILED: DNS problem")
        log(str(e))
        return False


def test_https():
    log("\n2) Testing HTTPS...")
    try:
        r = requests.get("https://" + DOMAIN, timeout=30)
        log(f"OK: HTTPS works. Status = {r.status_code}")
        return True
    except Exception as e:
        log("FAILED: HTTPS problem")
        log(str(e))
        return False


def submit_year_request(year):
    log("\n--------------------------------------")
    log(f"Submitting request for year {year}")
    log("--------------------------------------")

    # Important: API key is also included in URL query parameter
    url = BASE_URL + "?api_key=" + API_KEY

    payload = {
        "wkt": WKT,
        "attributes": ATTRIBUTES,
        "names": year,
        "interval": INTERVAL,
        # NSRDB records in UTC; utc=true keeps timestamps in UTC so episode
        # windows align exactly with the project time standard (no shifting).
        "utc": "true",
        "leap_day": "true",
        "email": EMAIL,
        "api_key": API_KEY,
    }

    headers = {
        "content-type": "application/x-www-form-urlencoded",
        "cache-control": "no-cache",
    }

    try:
        response = requests.post(
            url,
            data=payload,
            headers=headers,
            timeout=60
        )

        log(f"HTTP status: {response.status_code}")

        if response.status_code != 200:
            log("FAILED: Server returned error")
            log(response.text[:1000])
            return None

        data = response.json()

        errors = data.get("errors", [])
        if errors:
            log("FAILED: API returned errors:")
            for err in errors:
                log(f"- {err}")
            return None

        outputs = data.get("outputs", {})
        message = outputs.get("message")
        download_url = outputs.get("downloadUrl")

        log("OK: Request accepted")
        log(f"Message: {message}")

        if not download_url:
            log("FAILED: No downloadUrl found")
            log(str(data))
            return None

        log("OK: Download URL received")
        return download_url

    except Exception as e:
        log("FAILED: Could not submit request")
        log(str(e))
        return None


def wait_and_download_zip(download_url, year):
    log(f"\nWaiting for ZIP file for {year}...")

    max_attempts = 80
    wait_seconds = 15

    for attempt in range(1, max_attempts + 1):
        log(f"Attempt {attempt}/{max_attempts}")

        try:
            r = requests.get(download_url, timeout=120)

            log(f"Status: {r.status_code}")
            log(f"Content-Type: {r.headers.get('Content-Type', '')}")

            if r.status_code == 403:
                log("File is not ready yet. Waiting...")
                time.sleep(wait_seconds)
                continue

            if r.status_code != 200:
                log("Download failed. Response:")
                log(r.text[:1000])
                time.sleep(wait_seconds)
                continue

            content = r.content

            if content[:2] == b"PK":
                log("OK: Downloaded ZIP file")

                os.makedirs(OUTPUT_FOLDER, exist_ok=True)

                zip_path = os.path.join(
                    OUTPUT_FOLDER,
                    f"newyork_nsrdb_5min_{year}.zip"
                )

                with open(zip_path, "wb") as f:
                    f.write(content)

                log(f"Saved ZIP: {zip_path}")
                return content

            log("FAILED: Downloaded file is not ZIP")
            log(f"First 100 bytes: {content[:100]}")
            return None

        except Exception as e:
            log("Download error:")
            log(str(e))
            log("Waiting...")
            time.sleep(wait_seconds)

    log(f"FAILED: ZIP file for {year} was not ready")
    return None


def read_zip_to_dataframe(zip_bytes, year):
    log(f"\nExtracting and reading CSV for {year}...")

    try:
        year_folder = os.path.join(OUTPUT_FOLDER, year)
        # Remove any previous extraction so exactly one SAM CSV per year remains
        # (the prepare script picks the SAM file from this folder).
        if os.path.isdir(year_folder):
            import shutil
            shutil.rmtree(year_folder)
        os.makedirs(year_folder, exist_ok=True)

        z = zipfile.ZipFile(io.BytesIO(zip_bytes))

        log("Files inside ZIP:")
        for name in z.namelist():
            log(f"- {name}")

        z.extractall(year_folder)
        log(f"Extracted to: {year_folder}")

        csv_files = [
            name for name in z.namelist()
            if name.lower().endswith(".csv")
        ]

        if not csv_files:
            log("FAILED: No CSV file found inside ZIP")
            return None

        csv_name = csv_files[0]
        log(f"Reading CSV: {csv_name}")

        with z.open(csv_name) as f:
            # NSRDB CSV normally has 2 metadata rows before the table
            df = pd.read_csv(f, skiprows=2)

        df["source_year"] = year

        log("OK: pandas loaded the data")
        log(f"Rows: {len(df)}")
        log(f"Columns: {list(df.columns)}")

        clean_path = os.path.join(
            OUTPUT_FOLDER,
            f"newyork_nsrdb_5min_{year}_clean.csv"
        )

        df.to_csv(clean_path, index=False)
        log(f"Saved clean CSV: {clean_path}")

        log("First 5 rows:")
        print(df.head(), flush=True)

        return df

    except Exception as e:
        log("FAILED: Could not read ZIP/CSV")
        log(str(e))
        return None


def main():
    log("======================================")
    log("NEW YORK NSRDB 5-MIN MULTI-YEAR LOADER")
    log("======================================")
    log(f"WKT location: {WKT}")
    log(f"Years: {YEARS}")
    log(f"Interval: {INTERVAL} minutes")
    log(f"Output folder: {OUTPUT_FOLDER}")

    if API_KEY == "PUT_YOUR_API_KEY_HERE":
        log("\nERROR: Put your real API key in API_KEY.")
        return

    if EMAIL == "PUT_YOUR_EMAIL_HERE":
        log("\nERROR: Put your real email in EMAIL.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if not test_dns():
        log("STOP: DNS failed.")
        return

    if not test_https():
        log("STOP: HTTPS failed.")
        return

    all_dataframes = []

    for year in YEARS:
        log("\n\n======================================")
        log(f"STARTING YEAR {year}")
        log("======================================")

        download_url = submit_year_request(year)

        if not download_url:
            log(f"Skipping year {year}: request failed.")
            continue

        zip_bytes = wait_and_download_zip(download_url, year)

        if not zip_bytes:
            log(f"Skipping year {year}: download failed.")
            continue

        df = read_zip_to_dataframe(zip_bytes, year)

        if df is None:
            log(f"Skipping year {year}: pandas load failed.")
            continue

        all_dataframes.append(df)

        # Small pause to be polite with API
        log("Waiting 3 seconds before next year...")
        time.sleep(3)

    log("\n======================================")
    log("ALL YEAR REQUESTS FINISHED")
    log("======================================")

    if not all_dataframes:
        log("FAILED: No years were loaded successfully.")
        return

    log("Combining all years...")

    combined = pd.concat(all_dataframes, ignore_index=True)

    combined_path = os.path.join(
        OUTPUT_FOLDER,
        "newyork_nsrdb_5min_2018_2024_combined.csv"
    )

    combined.to_csv(combined_path, index=False)

    log("\nSUCCESS")
    log(f"Combined rows: {len(combined)}")
    log(f"Saved combined CSV: {combined_path}")

    log("\nFiles saved in:")
    log(OUTPUT_FOLDER)

    log("\nFirst 5 combined rows:")
    print(combined.head(), flush=True)


if __name__ == "__main__":
    main()