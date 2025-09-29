#python main_pollution.py
import schedule
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import numpy as np
import math
from location_data_updated import LOCATIONS

# ========== CONFIG ==========
API_KEY = "da23d8525160416db7e103801240910"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

DB_HOST = "postgis"
DB_PORT = "5432"
DB_NAME = "traffic_pollution_db"
DB_USER = "postgres"
DB_PASS = "1234"

RAW_FIELDS = ["raw_pm25","raw_pm10","raw_no2","raw_o3","raw_so2","raw_co",
              "raw_us_epa_index","raw_temperature_c","raw_humidity_percent",
              "raw_wind_speed_kmh","raw_precipitation_mm"]

INTERP_FIELDS = ["pm25","pm10","no2","o3","so2","co",
                 "us_epa_index","temperature_c","humidity_percent",
                 "wind_speed_kmh","precipitation_mm"]

IDW_POWER = 2
K_NEAREST = 12
EPS_DISTANCE = 1e-10

# ========== DB helper ==========
def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

def insert_pollution_data(records):
    sql = f"""
    INSERT INTO api.pollution_data
      (date, time, station_id,
       latitude_cctv, longitude_cctv,
       latitude_api, longitude_api,
       {",".join(RAW_FIELDS)})
    VALUES %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        execute_values(cur, sql, records)
        conn.commit()

def fetch_pollution_data(date_val, time_val):
    sql = f"""
    SELECT id, latitude_api, longitude_api,
           {",".join(RAW_FIELDS)}
    FROM api.pollution_data
    WHERE date = %s AND time = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (date_val, time_val))
        return cur.fetchall()

def fetch_cctv_locations(date_val, time_val):
    sql = """
    SELECT id, station_id, latitude_cctv, longitude_cctv
    FROM api.pollution_data
    WHERE date = %s AND time = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, (date_val, time_val))
        return cur.fetchall()

def update_interpolated(records):
    set_clause = ",".join([f"{f} = %s" for f in INTERP_FIELDS])
    sql = f"""
    UPDATE api.pollution_data
    SET {set_clause}
    WHERE id = %s
    """
    with get_connection() as conn, conn.cursor() as cur:
        cur.executemany(sql, records)
        conn.commit()

# ========== Helper ==========
def clean_val(v):
    if v is None:
        return None
    try:
        v = float(v)
        if math.isnan(v):
            return None
        return v
    except Exception:
        return None

# ===== Haversine distance (km) =====
def haversine(lon1, lat1, lon2, lat2):
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2.0)**2
    return R * 2.0 * np.arcsin(np.sqrt(a))

# ===== IDW predict (เวอร์ชันถูกต้อง) =====
def idw_predict(x_air, y_air, z_air, x0, y0, power=IDW_POWER, k=None):
    mask = ~np.isnan(z_air)
    x_air, y_air, z_air = x_air[mask], y_air[mask], z_air[mask]
    if len(z_air) == 0:
        return np.nan

    d = haversine(x_air, y_air, x0, y0)

    near0 = np.isclose(d, 0.0)
    if near0.any():
        return float(np.nanmean(z_air[near0]))

    if (k is not None) and (k < len(d)):
        idx = np.argpartition(d, k)[:k]
        d, z_air = d[idx], z_air[idx]

    w = 1.0 / np.maximum(d, EPS_DISTANCE)**power
    return float(np.sum(w * z_air) / np.sum(w))

# ========== Workflow ==========
def fetch_and_store():
    now = datetime.now()
    date_val = now.date()
    time_val = (now - timedelta(hours=1)).strftime("%H:%M:%S")
    print(f"⏳ Fetching API data: {date_val} {time_val}")

    raw_records = []

    for loc in LOCATIONS:
        station_id = loc["station_id"]
        lat_cctv, lon_cctv = loc["lat"], loc["lon"]

        params = {"key": API_KEY, "q": f"{lat_cctv},{lon_cctv}", "aqi": "yes"}

        rec = {
            "date": date_val,
            "time": time_val,
            "station_id": station_id,
            "lat_cctv": lat_cctv,
            "lon_cctv": lon_cctv,
            "lat_api": None,
            "lon_api": None,
        }
        for f in RAW_FIELDS:
            rec[f] = None  # ค่า default = NULL

        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            loc_api = data.get("location", {}) or {}
            rec["lat_api"] = loc_api.get("lat")
            rec["lon_api"] = loc_api.get("lon")

            current = data.get("current", {}) or {}
            aq = current.get("air_quality", {}) or {}

            rec.update({
                "raw_co": aq.get("co"),
                "raw_no2": aq.get("no2"),
                "raw_o3": aq.get("o3"),
                "raw_so2": aq.get("so2"),
                "raw_pm25": aq.get("pm2_5"),
                "raw_pm10": aq.get("pm10"),
                "raw_us_epa_index": aq.get("us-epa-index"),
                "raw_temperature_c": current.get("temp_c"),
                "raw_humidity_percent": current.get("humidity"),
                "raw_wind_speed_kmh": current.get("wind_kph"),
                "raw_precipitation_mm": current.get("precip_mm"),
            })

            print(f"✅ Success for {station_id}")
        except Exception as e:
            print(f"⚠️ Failed for {station_id}, inserted NULL row: {e}")

        raw_records.append(rec)

    # ===== Insert DB =====
    insert_records = [(
        r["date"], r["time"], r["station_id"],
        r["lat_cctv"], r["lon_cctv"], r["lat_api"], r["lon_api"],
        r["raw_pm25"], r["raw_pm10"], r["raw_no2"], r["raw_o3"],
        r["raw_so2"], r["raw_co"], r["raw_us_epa_index"],
        r["raw_temperature_c"], r["raw_humidity_percent"],
        r["raw_wind_speed_kmh"], r["raw_precipitation_mm"],
    ) for r in raw_records]

    insert_pollution_data(insert_records)
    print(f"✅ Inserted {len(insert_records)} rows into api.pollution_data")

    # ===== Interpolation =====
    stations = fetch_pollution_data(date_val, time_val)
    cameras = fetch_cctv_locations(date_val, time_val)
    if not stations or not cameras:
        print("⚠️ Missing data for interpolation")
        return

    update_rows = []
    for cam in cameras:
        row_id, cam_id, lat_cctv, lon_cctv = cam
        if lat_cctv is None or lon_cctv is None:
            continue

        x0, y0 = float(lon_cctv), float(lat_cctv)

        interp_values = []
        for j, out_col in enumerate(INTERP_FIELDS):
            raw_col_index = 3 + j
            z_air = np.array([s[raw_col_index] for s in stations], dtype=float)
            x_air = np.array([s[2] for s in stations], dtype=float)
            y_air = np.array([s[1] for s in stations], dtype=float)

            val = idw_predict(x_air, y_air, z_air, x0, y0,
                              power=IDW_POWER, k=K_NEAREST)
            interp_values.append(clean_val(val))

        update_rows.append((*interp_values, row_id))

    update_interpolated(update_rows)
    print(f"✅ Updated {len(update_rows)} rows with IDW results")


    # 5) Interpolation ด้วย IDW แบบใหม่
    stations = fetch_pollution_data(date_val, time_val)
    cameras = fetch_cctv_locations(date_val, time_val)
    if not stations or not cameras:
        print("⚠️ Missing data for interpolation")
        return

    update_rows = []
    for cam in cameras:
        row_id, cam_id, lat_cctv, lon_cctv = cam
        if lat_cctv is None or lon_cctv is None:
            continue

        x0, y0 = float(lon_cctv), float(lat_cctv)

        interp_values = []
        for j, out_col in enumerate(INTERP_FIELDS):
            raw_col_index = 3 + j  # เพราะ index 0=id,1=lat_api,2=lon_api,3=raw_pm25,...
            z_air = np.array([s[raw_col_index] for s in stations], dtype=float)
            x_air = np.array([s[2] for s in stations], dtype=float)
            y_air = np.array([s[1] for s in stations], dtype=float)

            val = idw_predict(x_air, y_air, z_air, x0, y0, power=IDW_POWER, k=K_NEAREST)
            interp_values.append(clean_val(val))

        update_rows.append((*interp_values, row_id))

    update_interpolated(update_rows)
    print(f"✅ Updated {len(update_rows)} rows with IDW results")

# ========== Run ==========
if __name__ == "__main__":
    # รันจริง 09:00, 13:00, 18:00
    schedule.every().day.at("09:00").do(fetch_and_store)
    schedule.every().day.at("13:00").do(fetch_and_store)
    schedule.every().day.at("18:00").do(fetch_and_store)

    print("⏳ รอเวลา 09:00, 13:00, 18:00 เพื่อรัน fetch_and_store() ...")
    while True:
        schedule.run_pending()
        time.sleep(1)
