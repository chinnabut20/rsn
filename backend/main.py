from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime, timedelta
from typing import Optional
import csv
import numpy as np
import pandas as pd
import os
import sys
import warnings
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanSquaredError as MSE
import joblib, pickle
from pydantic import BaseModel
import requests

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
# เก็บโมเดล + scaler + config
# ==========================
lstm_models = {}
scalers = {}
configs = {}

pollution_cols = ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"]
weather_cols = ["Temperature (C)", "Humidity (%)", "Wind Speed (km/h)", "Precipitation (mm)"]
traffic_cols = ["bus", "car", "motorcycle", "truck", "van"]

# ==========================
# โหลดโมเดลทั้งหมด
# ==========================
def load_all_models():
    model_dir = "models"
    
    if not os.path.exists(model_dir):
        print(f"❌ Model directory '{model_dir}' not found!")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Files in current directory: {os.listdir('.')}")
        return
    
    for p in pollution_cols:
        try:
            model_path = os.path.join(model_dir, f"{p}_model_lstm.h5")
            scaler_path = os.path.join(model_dir, f"{p}_model_scaler.pkl")
            config_path = os.path.join(model_dir, f"{p}_model_config.pkl")
            
            if not os.path.exists(model_path): continue
            if not os.path.exists(scaler_path): continue
            if not os.path.exists(config_path): continue
            
            lstm_models[p] = load_model(
                model_path,
                custom_objects={
                    "mse": MeanSquaredError(),
                    "mean_squared_error": MSE(),
                    "MeanSquaredError": MSE()
                }
            )
            scalers[p] = joblib.load(scaler_path)
            with open(config_path, "rb") as f:
                configs[p] = pickle.load(f)
            
            print(f"✅ Successfully loaded {p} model")
        except Exception as e:
            print(f"⚠️ Failed to load model for {p}: {e}")



@app.get("/model_status")
async def model_status():
    status = {}
    for pollutant in pollution_cols:
        status[pollutant] = {
            "model_loaded": pollutant in lstm_models,
            "scaler_loaded": pollutant in scalers,
            "config_loaded": pollutant in configs
        }
    return {"model_status": status}

# โหลดโมเดลตอน startup
try:
    load_all_models()
    print(f"🚀 FastAPI started. Loaded {len(lstm_models)} models out of {len(pollution_cols)}")
except Exception as e:
    print(f"❌ Error during model loading: {e}")

# ==========================
# Schema
# ==========================
class PredictRequest(BaseModel):
    pollutant: str
    recent_data: list

# ==========================
# ฟังก์ชันเตรียมข้อมูล
# ==========================
def prepare_daily_df():
    try:
        pollution_file = "cctv_pollution_IDW_with_station_id.csv"
        vehicle_file = "combined_file.csv"
        
        if not os.path.exists(pollution_file):
            raise FileNotFoundError(f"Pollution data file not found: {pollution_file}")
        if not os.path.exists(vehicle_file):
            raise FileNotFoundError(f"Vehicle data file not found: {vehicle_file}")
        
        pollution_df = pd.read_csv(pollution_file)
        vehicle_df = pd.read_csv(vehicle_file)

        merged_df = pd.merge(pollution_df, vehicle_df, on=["Date", "Time", "Station ID"], how="outer")
        merged_df["Date"] = pd.to_datetime(merged_df["Date"], dayfirst=True, errors="coerce")

        if "ImageCount" in merged_df.columns:
            for col in traffic_cols:
                if col in merged_df.columns:
                    mask = merged_df["ImageCount"] == 0
                    avg_station_time = merged_df.groupby(["Station ID","Time"])[col].transform("mean")
                    avg_station_day  = merged_df.groupby(["Station ID","Date"])[col].transform("mean")
                    avg_all_time     = merged_df.groupby("Time")[col].transform("mean")
                    merged_df.loc[mask, col] = (
                        avg_station_time[mask]
                        .fillna(avg_station_day[mask])
                        .fillna(avg_all_time[mask])
                    )

        def interpolate_pollution_weather(group):
            group = group.set_index("Date")
            existing_pollution_cols = [col for col in pollution_cols if col in group.columns]
            existing_weather_cols = [col for col in weather_cols if col in group.columns]
            if existing_pollution_cols:
                group[existing_pollution_cols] = group[existing_pollution_cols].interpolate(method="time", limit_direction="both").ffill().bfill()
            if existing_weather_cols:
                group[existing_weather_cols] = group[existing_weather_cols].interpolate(method="time", limit_direction="both").ffill().bfill()
            return group.reset_index()

        merged_df = merged_df.groupby("Station ID", group_keys=False).apply(interpolate_pollution_weather)
        existing_traffic_cols = [col for col in traffic_cols if col in merged_df.columns]
        if existing_traffic_cols:
            merged_df[existing_traffic_cols] = merged_df[existing_traffic_cols].round(0).astype("Int64")

        agg_cols = []
        for col_list in [pollution_cols, traffic_cols, weather_cols]:
            agg_cols.extend([col for col in col_list if col in merged_df.columns])
        
        df_daily = (
            merged_df.groupby(["Date","Station ID"])[agg_cols]
            .mean(numeric_only=True)
            .reset_index()
        )
        return df_daily
    
    except Exception as e:
        print(f"Error in prepare_daily_df: {e}")
        raise

# ==========================
# ฟังก์ชันทำนายทีละ station
# ==========================
def forecast_target_for_station(df_daily, station_id, target, days=7):
    if target not in lstm_models:
        raise ValueError(f"Model for {target} not loaded!")
    
    model = lstm_models[target]
    scaler = scalers[target]
    config = configs[target]
    features = config.get("features", traffic_cols + weather_cols + [target])
    seq_length = config.get("seq_length", 7)

    g = df_daily[df_daily["Station ID"] == station_id].sort_values("Date").copy()
    if g.shape[0] < seq_length:
        raise ValueError(f"Not enough history for station {station_id}")
    
    for f in features:
        if f not in g.columns:
            g[f] = 0.0

    g_feat = g[features].iloc[-seq_length:].astype(float).fillna(0.0)
    g_feat_scaled = scaler.transform(g_feat.values)
    current_seq = g_feat_scaled.reshape(1, seq_length, len(features))
    target_idx = features.index(target)
    
    forecasts = []
    for step in range(days):
        next_pred_scaled = model.predict(current_seq, verbose=0)[0, 0]
        dummy = np.zeros((1, len(features)))
        dummy[0, target_idx] = next_pred_scaled
        next_pred = scaler.inverse_transform(dummy)[0, target_idx]
        forecasts.append(float(next_pred))
        new_row = current_seq[0, -1, :].copy()
        new_row[target_idx] = next_pred_scaled
        current_seq = np.roll(current_seq, -1, axis=1)
        current_seq[0, -1, :] = new_row
    return forecasts

# ==========================
# Forecast endpoint หลัก
# ==========================
@app.get("/forecast_pollution")
async def forecast_pollution(
    station_id: Optional[str] = Query(None, description="Station ID"),
    targets: Optional[str] = Query(None, description="Target pollutant(s), comma separated"),
    days: int = Query(7, description="Number of days to forecast")
):
    try:
        if not lstm_models:
            raise HTTPException(status_code=500, detail="No models loaded!")
        
        df_daily = prepare_daily_df()
        stations = [station_id] if station_id else sorted(df_daily["Station ID"].unique().tolist())

        # ✅ ใช้ targets ถ้ามี
        if targets:
            requested = [t.strip() for t in targets.split(",")]
            targets_list = [p for p in requested if p in lstm_models]
        else:
            targets_list = [p for p in pollution_cols if p in lstm_models]

        result = []
        last_date = df_daily["Date"].max()
        
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
                day_prediction = {"date": forecast_date, "day": day_idx + 1, "pollutants": {}}
                for pollutant in targets_list:
                    forecasts = station_forecasts[pollutant]
                    value = round(float(forecasts[day_idx]), 3) if len(forecasts) > day_idx else 0.0
                    day_prediction["pollutants"][pollutant] = value
                station_entry["predictions"].append(day_prediction)
            result.append(station_entry)

        return {"status": "success", "total_stations": len(result), "forecast_period": f"{days} days", "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forecast failed: {str(e)}")

# ==========================
# Forecast endpoint แยก
# ==========================
@app.get("/forecast_pm25")
async def forecast_pm25(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="PM2.5", days=days)

@app.get("/forecast_pm10")
async def forecast_pm10(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="PM10", days=days)

@app.get("/forecast_no2")
async def forecast_no2(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="NO2", days=days)

@app.get("/forecast_so2")
async def forecast_so2(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="SO2", days=days)

@app.get("/forecast_o3")
async def forecast_o3(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="O3", days=days)

@app.get("/forecast_co")
async def forecast_co(station_id: Optional[str] = Query(None), days: int = Query(7)):
    return await forecast_pollution(station_id=station_id, targets="CO", days=days)

# ==========================
# Utility: แปลงวันที่
# ==========================
def convert_date_format(react_date: str) -> str:
    try:
        date_obj = datetime.strptime(react_date, "%Y-%m-%d")
        return f"{date_obj.day}/{date_obj.month}/{date_obj.year}"
    except ValueError:
        return react_date

# ==========================
# API: Vehicle counts
# ==========================
@app.get("/vehicle_counts")
async def get_vehicle_counts(date: Optional[str] = Query(None), time: Optional[str] = Query(None)):
    vehicle_counts = []
    csv_date_format = convert_date_format(date) if date else None
    try:
        if not os.path.exists("combined_file.csv"):
            raise FileNotFoundError("combined_file.csv not found")
        with open("combined_file.csv", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if csv_date_format and row["Date"] != csv_date_format: continue
                if time and row["Time"] != time: continue
                vehicle_counts.append({
                    "Date": row["Date"], 
                    "Time": row["Time"], 
                    "Station ID": row["Station ID"],
                    "bus": int(row.get("bus", 0)) if row.get("bus") else 0,
                    "car": int(row.get("car", 0)) if row.get("car") else 0,
                    "motorcycle": int(row.get("motorcycle", 0)) if row.get("motorcycle") else 0,
                    "truck": int(row.get("truck", 0)) if row.get("truck") else 0,
                    "van": int(row.get("van", 0)) if row.get("van") else 0
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading vehicle data: {str(e)}")
    return {"data": vehicle_counts}

# ==========================
# API: Pollution data
# ==========================
@app.get("/pollution_data")
async def get_pollution_data(date: Optional[str] = Query(None), time: Optional[str] = Query(None)):
    pollution_data = []
    csv_date_format = convert_date_format(date) if date else None
    try:
        if not os.path.exists("cctv_pollution_IDW_with_station_id.csv"):
            raise FileNotFoundError("cctv_pollution_IDW_with_station_id.csv not found")
        with open("cctv_pollution_IDW_with_station_id.csv", "r", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if csv_date_format and row["Date"] != csv_date_format: continue
                if time and row["Time"] != time: continue
                pollution_data.append({
                    "Date": row["Date"], 
                    "Time": row["Time"], 
                    "Station ID": row["Station ID"],
                    "CO": float(row["CO"]) if row.get("CO") and row["CO"] != '' else None,
                    "NO2": float(row["NO2"]) if row.get("NO2") and row["NO2"] != '' else None,
                    "O3": float(row["O3"]) if row.get("O3") and row["O3"] != '' else None,
                    "SO2": float(row["SO2"]) if row.get("SO2") and row["SO2"] != '' else None,
                    "PM2.5": float(row["PM2.5"]) if row.get("PM2.5") and row["PM2.5"] != '' else None,
                    "PM10": float(row["PM10"]) if row.get("PM10") and row["PM10"] != '' else None,
                    "Temperature (C)": float(row["Temperature (C)"]) if row.get("Temperature (C)") and row["Temperature (C)"] != '' else None,
                    "Humidity (%)": float(row["Humidity (%)"]) if row.get("Humidity (%)") and row["Humidity (%)"] != '' else None,
                    "Wind Speed (km/h)": float(row["Wind Speed (km/h)"]) if row.get("Wind Speed (km/h)") and row["Wind Speed (km/h)"] != '' else None,
                    "Precipitation (mm)": float(row["Precipitation (mm)"]) if row.get("Precipitation (mm)") and row["Precipitation (mm)"] != '' else None,
                })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading pollution data: {str(e)}")
    return {"data": pollution_data}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
