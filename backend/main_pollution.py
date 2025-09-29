# python main_pollution.py
import schedule
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import numpy as np
import math
from location_data_updated import LOCATIONS
from zoneinfo import ZoneInfo

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

# ===== IDW predict =====
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
    now = datetime.now(ZoneInfo("Asia/Bangkok"))   # ใช้เวลาไทยในการประมวลผล
    date_val = now.date()
    time_val = (now - timedelta(hours=1)).strftime("%H:%M:%S")
    print(f"⏳ Fetching API data: {date_val} {time_val} (เวลาไทย)")

    # ... (ส่วนเดิมของโค้ด fetch, insert, interpolate) ...

# ========== Run ==========
if __name__ == "__main__":
    # รันจริงตามเวลาไทย → ต้องแปลงเป็นเวลา UTC สำหรับ schedule
    schedule.every().day.at("02:00").do(fetch_and_store)  # 09:00 ไทย
    schedule.every().day.at("06:00").do(fetch_and_store)  # 13:00 ไทย
    schedule.every().day.at("11:00").do(fetch_and_store)  # 18:00 ไทย

    print("⏳ รอเวลา 09:00, 13:00, 18:00 (เวลาไทย) เพื่อรัน fetch_and_store() ...")
    while True:
        schedule.run_pending()
        time.sleep(1)
