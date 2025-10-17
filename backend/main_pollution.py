import schedule
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from datetime import datetime, timedelta
import numpy as np
import math
import pytz
from location_data_updated import LOCATIONS

# ========== CONFIG ==========
API_KEY = "da23d8525160416db7e103801240910"
BASE_URL = "http://api.weatherapi.com/v1/current.json"

DB_HOST = "postgis"
DB_PORT = "5432"
DB_NAME = "traffic_pollution_db"
DB_USER = "postgres"
DB_PASS = "1234"

RAW_FIELDS = [
    "raw_pm25","raw_pm10","raw_no2","raw_o3","raw_so2","raw_co",
    "raw_us_epa_index","raw_temperature_c","raw_humidity_percent",
    "raw_wind_speed_kmh","raw_precipitation_mm"
]

INTERP_FIELDS = [
    "pm25","pm10","no2","o3","so2","co",
    "us_epa_index","temperature_c","humidity_percent",
    "wind_speed_kmh","precipitation_mm"
]

# ตั้งเวลาให้ตรงกับ Timezone ของประเทศไทย
bangkok_tz = pytz.timezone('Asia/Bangkok')

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

def mean_ignore_none(values):
    nums = [float(v) for v in values if v is not None]
    return (sum(nums)/len(nums)) if nums else None

# ========== IDW ==========
def idw_interpolation(xy_train, values, xy_pred, power=2):
    result = []
    for x, y in xy_pred:
        d = np.sqrt((xy_train[:,0]-x)**2 + (xy_train[:,1]-y)**2)
        d = np.where(d == 0, 1e-10, d)
        w = 1 / d**power
        result.append(np.sum(w*values)/np.sum(w))
    return np.array(result)

# ========== Workflow ==========
def fetch_and_store():
    now = datetime.now(bangkok_tz)
    date_val = now.date()
    time_val = (now - timedelta(hours=1)).strftime("%H:%M:%S")
    print(f"⏳ Fetching API data: {date_val} {time_val}")

    raw_records = []
    for loc in LOCATIONS:
        lat_cctv, lon_cctv = loc["lat"], loc["lon"]
        station_id = loc["station_id"]

        params = {"key": API_KEY, "q": f"{lat_cctv},{lon_cctv}", "aqi": "yes"}
        try:
            resp = requests.get(BASE_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()

            loc_api = data.get("location", {}) or {}
            lat_api, lon_api = loc_api.get("lat"), loc_api.get("lon")

            current = data.get("current", {}) or {}
            aq = current.get("air_quality", {}) or {}

            rec = {
                "date": date_val,
                "time": time_val,
                "station_id": station_id,
                "lat_cctv": lat_cctv,
                "lon_cctv": lon_cctv,
                "lat_api": lat_api,
                "lon_api": lon_api,
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
            }
            raw_records.append(rec)
        except Exception as e:
            print(f"❌ Error for {station_id}: {e}")
            rec = {f: None for f in RAW_FIELDS}
            rec.update({
                "date": date_val,
                "time": time_val,
                "station_id": station_id,
                "lat_cctv": lat_cctv,
                "lon_cctv": lon_cctv,
                "lat_api": None,
                "lon_api": None,
            })
            raw_records.append(rec)

    if not raw_records:
        print("⚠️ ไม่มีข้อมูลจาก API")
        return

    # 2) เติมค่าเฉลี่ยสำหรับค่าที่ขาดหาย
    batch_means = {f: mean_ignore_none([r[f] for r in raw_records]) for f in RAW_FIELDS}

    station_hourly_cache = {}
    global_hourly_means = {}
    global_means = {}

    hour_val = int(time_val[:2])  # ใช้ชั่วโมงแทน LIKE

    with get_connection() as conn, conn.cursor() as cur:
        # ค่าเฉลี่ยของทุกสถานีในช่วงเวลาเดียวกัน 14 วันล่าสุด
        cur.execute(f"""
            SELECT {",".join([f"AVG({f})" for f in RAW_FIELDS])}
            FROM api.pollution_data
            WHERE EXTRACT(HOUR FROM time) = %s
              AND date >= CURRENT_DATE - INTERVAL '14 days'
        """, (hour_val,))
        row = cur.fetchone()
        if row:
            global_hourly_means = dict(zip(RAW_FIELDS, row))

        # ค่าเฉลี่ยทั่วทั้งหมดใน 30 วันล่าสุด
        cur.execute(f"""
            SELECT {",".join([f"AVG({f})" for f in RAW_FIELDS])}
            FROM api.pollution_data
            WHERE date >= CURRENT_DATE - INTERVAL '30 days'
        """)
        row = cur.fetchone()
        if row:
            global_means = dict(zip(RAW_FIELDS, row))

    imputed_count = 0
    for rec in raw_records:
        sid = rec["station_id"]
        if sid not in station_hourly_cache:
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(f"""
                    SELECT {",".join([f"AVG({f})" for f in RAW_FIELDS])}
                    FROM api.pollution_data
                    WHERE station_id = %s
                      AND EXTRACT(HOUR FROM time) = %s
                      AND date >= CURRENT_DATE - INTERVAL '14 days'
                """, (sid, hour_val))
                row = cur.fetchone()
                if row:
                    station_hourly_cache[sid] = dict(zip(RAW_FIELDS, row))
                else:
                    station_hourly_cache[sid] = {}

        for f in RAW_FIELDS:
            if rec[f] is None:
                val = batch_means.get(f)
                if val is None:
                    val = station_hourly_cache[sid].get(f)
                if val is None:
                    val = global_hourly_means.get(f)
                if val is None:
                    val = global_means.get(f)
                if val is None:
                    val = 0.0
                rec[f] = float(val)
                imputed_count += 1

    print(f"✨ เติมค่าที่ขาดหายด้วยการเฉลี่ยสำเร็จ: {imputed_count} ช่อง")

    # 3) Insert DB
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

    # 4) Interpolation (เฉพาะครั้งเดียว)
    stations = fetch_pollution_data(date_val, time_val)
    cameras = fetch_cctv_locations(date_val, time_val)
    if not stations or not cameras:
        print("⚠️ Missing data for interpolation")
        return

    xy_train = np.array([[s[2], s[1]] for s in stations])  # (lon,lat)
    update_rows = []

    for cam in cameras:
        row_id, cam_id, lat_cctv, lon_cctv = cam
        xy_pred = np.array([[lon_cctv, lat_cctv]])

        interp_values = []
        for j in range(3, 3+len(RAW_FIELDS)):  # index 3 = raw_pm25
            values = np.array([s[j] for s in stations if s[j] is not None])
            xy_valid = np.array([[s[2], s[1]] for s in stations if s[j] is not None])
            if len(values) == 0:
                val = None
            else:
                val = idw_interpolation(xy_valid, values, xy_pred, power=2)[0]
            interp_values.append(clean_val(val))

        update_rows.append((*interp_values, row_id))

    update_interpolated(update_rows)
    print(f"✅ Updated {len(update_rows)} rows with IDW results")

# ========== Scheduler ==========
def run_scheduler():
    print("🚀 Scheduler started for pollution data fetching (Bangkok time)")
    print("🕒 Waiting for schedule times: 09:00, 13:00, 18:00 ...\n")

    while True:
        now = datetime.now(bangkok_tz)
        current_time = now.strftime("%H:%M")

        # รันตอนเวลาที่กำหนด
        if current_time in ["09:00", "13:00", "18:00", "23:21", "23:22"]:
            print(f"🕘 Running fetch_and_store at {current_time} (Bangkok)")
            fetch_and_store()
            time.sleep(60)  # ป้องกันการรันซ้ำในนาทีเดียวกัน

        time.sleep(1)

# ========== Run ==========
if __name__ == "__main__":
    run_scheduler()
