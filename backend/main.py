# uvicorn main:app --reload
import requests
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime, timedelta, date
from typing import Optional
import numpy as np
import pandas as pd
import os
import warnings
import psycopg2
import joblib, pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
from tensorflow.keras import layers

# ==========================
# FastAPI Setup
# ==========================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# DB Config
# ==========================
DB_HOST = "postgis"
DB_PORT = "5432"
DB_NAME = "traffic_pollution_db"
DB_USER = "postgres"
DB_PASS = "1234"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT,
        dbname=DB_NAME, user=DB_USER, password=DB_PASS
    )

# ==========================
# Model config
# ==========================
lstm_models = {}
scalers = {}
configs = {}

pollution_cols = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]
weather_cols = ["Temperature (C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (mm)"]
traffic_cols = ["bus", "car", "motorcycle", "truck", "van"]

# ==========================
# Load models
# ==========================
def load_all_models():
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # model_dir = os.path.join(BASE_DIR, "models")
    model_dir = "./models"
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found")
        return

    for p in pollution_cols:
        try:
            model_path = os.path.join(model_dir, f"{p}_model_lstm.h5")
            scaler_path = os.path.join(model_dir, f"{p}_model_scaler.pkl")
            config_path = os.path.join(model_dir, f"{p}_model_config.pkl")

            print(f"Loading model for {p} from {model_path}, {scaler_path}, {config_path}")

            if not (os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(config_path)):
                print(f"⚠️ Missing files for {p}")
                continue

            # lstm_models[p] = load_model(
            #     model_path,
            #     custom_objects={
            #         "mse": MeanSquaredError(),
            #         "mean_squared_error": MSE(),
            #         "MeanSquaredError": MSE()
            #     }
            # )


            # เพิ่มใน load_all_models():
            lstm_models[p] = load_model(
                model_path,
                compile=False,  # ❗ ปิดการ compile ตอนโหลด
                custom_objects={
                    "mse": MeanSquaredError(),
                    "mean_squared_error": MSE(),
                    "MeanSquaredError": MSE(),
                    "InputLayer": layers.InputLayer  # ป้องกัน error 'batch_shape'
                }
            )

            scalers[p] = joblib.load(scaler_path)
            with open(config_path, "rb") as f:
                configs[p] = pickle.load(f)

            print(f"✅ Loaded {p} model")

        except Exception as e:
            print(f"❌ Error loading {p}: {e}")

try:
    load_all_models()
    print(f"🚀 FastAPI started. Loaded {len(lstm_models)} models")
except Exception as e:
    print(f"❌ Error during model loading: {e}")

def prepare_daily_df():
    try:
        import requests, pandas as pd
        from datetime import timedelta

        pollution_url = "https://geodev.fun/rsn_api/pollution_data"
        vehicle_url   = "https://geodev.fun/rsn_api/vehicle_counts"

        pollution_df = pd.DataFrame(requests.get(pollution_url).json().get("data", []))
        traffic_df   = pd.DataFrame(requests.get(vehicle_url).json().get("data", []))

        # === Merge ===
        merged_df = pd.merge(
            pollution_df, traffic_df,
            on=["date", "time", "station_id"],
            how="outer"
        )

        # === เตรียม column และแปลงชนิดข้อมูล ===
        merged_df["date"] = pd.to_datetime(merged_df["date"], errors="coerce")
        merged_df = merged_df.sort_values(["station_id", "date", "time"]).reset_index(drop=True)

        traffic_cols = ["bus", "car", "motorcycle", "truck", "van", "total_vehicles"]
        pollution_cols = ["co", "no2", "o3", "so2", "pm25", "pm10"]
        weather_cols = ["temperature_c", "humidity_percent", "wind_speed_kmh", "precipitation_mm"]

        # ============================================
        # เติมข้อมูลรถเมื่อ “ไม่มีข้อมูล” (ทุกประเภทเป็นศูนย์)
        # ============================================
        if all(col in merged_df.columns for col in traffic_cols):
            # mask วันที่รถทุกประเภทเป็น 0 → ถือว่าไม่มีข้อมูล
            zero_mask = (merged_df[traffic_cols].sum(axis=1) == 0)

            for col in traffic_cols:
                avg_station_time = merged_df.groupby(['station_id', 'time'])[col].transform('mean')
                avg_station_day  = merged_df.groupby(['station_id', 'date'])[col].transform('mean')
                avg_all_time     = merged_df.groupby('time')[col].transform('mean')

                merged_df.loc[zero_mask, col] = (
                    avg_station_time[zero_mask]
                    .fillna(avg_station_day[zero_mask])
                    .fillna(avg_all_time[zero_mask])
                )

        # ============================================
        # เติมข้อมูลมลพิษและอุตุนิยมวิทยา
        # ============================================
        def interpolate_pollution_weather(group):
            group = group.set_index("date")
            cols_pollution = [c for c in pollution_cols if c in group.columns]
            cols_weather = [c for c in weather_cols if c in group.columns]

            if cols_pollution:
                group[cols_pollution] = (
                    group[cols_pollution]
                    .interpolate(method="time", limit_direction="both")
                    .fillna(method="ffill").fillna(method="bfill")
                )
            if cols_weather:
                group[cols_weather] = (
                    group[cols_weather]
                    .interpolate(method="time", limit_direction="both")
                    .fillna(method="ffill").fillna(method="bfill")
                )

            return group.reset_index()

        merged_df = merged_df.groupby("station_id", group_keys=False).apply(interpolate_pollution_weather)

        # ============================================
        # ปัดค่ารถให้เป็น int
        # ============================================
        for col in traffic_cols:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].round(0).astype("Int64")

        # ============================================
        # ทำ daily average
        # ============================================
        agg_cols = [c for c in pollution_cols + traffic_cols + weather_cols if c in merged_df.columns]
        df_daily = (
            merged_df.groupby(["date", "station_id"])[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

        for col in traffic_cols:
            if col in df_daily.columns:
                df_daily[col] = df_daily[col].round(0).astype("Int64")

        print(f"✅ prepare_daily_df สำเร็จ: {len(df_daily)} แถว จาก {df_daily['station_id'].nunique()} สถานี")
        return df_daily

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"❌ prepare_daily_df error: {e}")
        return pd.DataFrame()


# ==========================
# Forecast functions
# ==========================
def forecast_target_for_station(df_daily, station_id, target, days=7):
    if target not in lstm_models:
        raise ValueError(f"Model for {target} not loaded")

    model = lstm_models[target]
    scaler = scalers[target]
    config = configs[target]
    features = config.get("features", traffic_cols + weather_cols + [target])
    seq_length = config.get("seq_length", 7)

    g = df_daily[df_daily["station_id"] == station_id].sort_values("date").copy()
    if g.shape[0] < seq_length:
        raise ValueError(f"Not enough history for {station_id}")

    for f in features:
        if f not in g.columns:
            g[f] = 0.0

    g_feat = g[features].iloc[-seq_length:].astype(float).fillna(0.0)
    warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
    g_feat_scaled = scaler.transform(g_feat.values)
    warnings.resetwarnings()

    current_seq = g_feat_scaled.reshape(1, seq_length, len(features))
    target_idx = features.index(target)
    forecasts = []

    for step in range(days):
        next_pred_scaled = model.predict(current_seq, verbose=0)[0, 0]
        dummy = np.zeros((1, len(features)))
        dummy[0, target_idx] = next_pred_scaled
        warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
        next_pred = scaler.inverse_transform(dummy)[0, target_idx]
        warnings.resetwarnings()
        forecasts.append(float(next_pred))

        new_row = current_seq[0, -1, :].copy()
        new_row[target_idx] = next_pred_scaled
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_row

    return forecasts

# ==========================
# Forecast endpoints
# ==========================
@app.get("/rsn_api/forecast_pollution")
async def forecast_pollution(
    station_id: Optional[str] = Query(None),
    days: int = Query(7)
):
    try:
        print(lstm_models)
        if not lstm_models:
            raise HTTPException(status_code=500, detail="No models loaded")

        df_daily = prepare_daily_df()
        stations = [station_id] if station_id else sorted(df_daily["station_id"].unique().tolist())
        targets_list = [p for p in pollution_cols if p in lstm_models]

        result = []
        last_date = df_daily["date"].max()

        for sid in stations:
            station_entry = {"station_id": sid, "forecast_days": days, "predictions": []}
            station_forecasts = {}
            for pollutant in targets_list:
                try:
                    station_forecasts[pollutant] = forecast_target_for_station(df_daily, sid, pollutant, days)
                except Exception:
                    station_forecasts[pollutant] = [0.0] * days

            for day_idx in range(days):
                forecast_date = (last_date + timedelta(days=day_idx+1)).strftime("%Y-%m-%d")
                day_prediction = {"date": forecast_date, "day": day_idx+1, "pollutants": {}}
                for pollutant in targets_list:
                    forecasts = station_forecasts[pollutant]
                    value = round(float(forecasts[day_idx]), 3) if len(forecasts) > day_idx else 0.0
                    day_prediction["pollutants"][pollutant] = value
                station_entry["predictions"].append(day_prediction)
            result.append(station_entry)

        return {"status": "success", "total_stations": len(result), "forecast_period": f"{days} days", "data": result}

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

# ==========================
# Vehicle counts API (DB)
# ==========================
@app.get("/rsn_api/vehicle_counts")
async def get_vehicle_counts(date: Optional[str] = Query(None), time: Optional[str] = Query(None)):
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
            SELECT date, time, station_id, bus, car, motorcycle, truck, van, total_vehicles, province
            FROM api.traffic_data
            WHERE 1=1
        """
        params = []
        if date:
            query += " AND date = %s"
            params.append(date)
        if time:
            query += " AND time = %s"
            params.append(time)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        data = [dict(zip(colnames, row)) for row in rows]
        cur.close(); conn.close()
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

# ==========================
# Pollution data API (DB)
# ==========================
@app.get("/rsn_api/pollution_data")
async def get_pollution_data(date: Optional[str] = Query(None), time: Optional[str] = Query(None)):
    try:
        conn = get_connection()
        cur = conn.cursor()

        query = """
            SELECT date, time, station_id,
                   co, no2, o3, so2, pm25, pm10,
                   temperature_c, humidity_percent, wind_speed_kmh, precipitation_mm
            FROM api.pollution_data
            WHERE 1=1
        """
        params = []
        if date:
            query += " AND date = %s"
            params.append(date)
        if time:
            query += " AND time = %s"
            params.append(time)

        cur.execute(query, tuple(params))
        rows = cur.fetchall()
        colnames = [desc[0] for desc in cur.description]
        data = [dict(zip(colnames, row)) for row in rows]
        cur.close(); conn.close()
        return {"data": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")
    

@app.get("/rsn_api/available_dates")
async def available_dates():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT MIN(date) FROM api.traffic_data;")
    min_date = cur.fetchone()[0]
    cur.close()
    conn.close()

    # ให้ max_date = วันนี้ แทนที่จะเอาจาก DB
    today = date.today()

    return {"min_date": str(min_date), "max_date": str(today)}

@app.get("/rsn_api/CCTV_locations")
async def get_cctv_locations():
    try:
        # get from geojson file
        print("Fetching CCTV locations from GeoJSON file")
        geojson_path = "./CCTV_locations.geojson"
        if not os.path.exists(geojson_path):
            raise HTTPException(status_code=404, detail="GeoJSON file not found")
        import json
        with open(geojson_path, "r", encoding="utf-8") as f:
            geojson_data = json.load(f)
        return geojson_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB error: {str(e)}")

# ==========================
# Run server
# ==========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
