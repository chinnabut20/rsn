#python3 cap_yolo_db1.py
import os
import time
import threading
import subprocess
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import imagehash
from PIL import Image
import filecmp
from url3 import urls
from ultralytics import YOLO
import psycopg2  
import shutil
import cv2 
import pytz


yolo_model = YOLO("best.pt")  

# ตั้งค่า DB
DB_HOST = "postgis"
DB_PORT = "5432"
DB_NAME = "traffic_pollution_db"
DB_USER = "postgres"
DB_PASS = "1234"

# ตั้งค่าการแคป
CAPTURE_INTERVAL_SEC = 60
NUM_CAPTURES = 60
CAPTURE_TIMES = [(8, 0), (12, 0), (17, 0)]

bangkok_tz = pytz.timezone("Asia/Bangkok")


# === แคปภาพ 1 ภาพด้วย OpenCV
def capture_image_opencv(output_path, url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        # print(f"❌ ไม่สามารถเปิด stream ได้: {url}")
        return False

    ret, frame = cap.read()
    cap.release()

    if ret and frame is not None:
        try:
            cv2.imwrite(output_path, frame)
            return True
        except Exception as e:
            # print(f"❌ บันทึกภาพไม่สำเร็จ: {e}")
            return False
    else:
        # print(f"⚠️ ดึงเฟรมไม่สำเร็จ: {url}")
        return False


# === ฟังก์ชันเชื่อมต่อ DB
def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS
    )


# === Insert ข้อมูล YOLO result ลง DB
def insert_to_db(data):
    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO api.traffic_data
            (date, time, station_id, imagecount, bus, car, motorcycle, truck, van, total_vehicles, province)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            data["date"], data["time"], data["station_id"], data["image_count"],
            data["bus"], data["car"], data["motorcycle"],
            data["truck"], data["van"], data["total_vehicles"], data["province"]
        ))
        conn.commit()
        cur.close()
        conn.close()
        # print(f"✅ Inserted to DB: {data['station_id']} {data['date']} {data['time']}")
    except Exception as e:
        print(f"❌ Failed to insert DB: {e}")


# === แคปภาพจากกล้องหนึ่งกล้อง
def run_camera_capture(province, cam_key, url, folder_name):
    start_time = datetime.now(bangkok_tz).replace(second=0, microsecond=0)

    for i in range(NUM_CAPTURES):
        target_time = start_time + timedelta(minutes=i)
        while datetime.now(bangkok_tz) < target_time:
            time.sleep(1)

        timestamp = target_time.strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(folder_name, province, cam_key)
        os.makedirs(save_dir, exist_ok=True)
        output_path = os.path.join(save_dir, f"{cam_key}_{timestamp}.jpg")

        retry_deadline = target_time + timedelta(seconds=30)
        while datetime.now(bangkok_tz) < retry_deadline:
            if capture_image_opencv(output_path, url):
                break

            time.sleep(2)

# === ลบภาพซ้ำและภาพเสียในโฟลเดอร์
# def cleanup_duplicate_and_bad_images(root_folder):
#     hash_dict = defaultdict(list)

#     for dirpath, _, filenames in os.walk(root_folder):
#         for filename in filenames:
#             if filename.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
#                 file_path = os.path.join(dirpath, filename)
#                 try:
#                     with Image.open(file_path) as img:
#                         img_hash = imagehash.average_hash(img)
#                         hash_dict[str(img_hash)].append(file_path)
#                 except Exception:
#                     try:
#                         os.remove(file_path)
#                     except:
#                         pass

#     for paths in hash_dict.values():
#         if len(paths) > 1:
#             for i in range(1, len(paths)):  # เก็บไฟล์แรก ลบไฟล์ถัดไป
#                 try:
#                     os.remove(paths[i])
#                 except:
#                     pass

def run_yolo_on_folder(folder_path, station_id, province, capture_time_str):
    counts = {"bus":0, "car":0, "motorcycle":0, "truck":0, "van":0}
    image_count = 0

    for file in os.listdir(folder_path):
        if file.lower().endswith((".jpg", ".png", ".jpeg", ".bmp")):
            image_count += 1
            img_path = os.path.join(folder_path, file)
            results = yolo_model(img_path, verbose=False)

            for r in results:
                for cls in r.boxes.cls:
                    cls_name = yolo_model.names[int(cls)]
                    if cls_name in counts:
                        counts[cls_name] += 1
    
    total = sum(counts.values())
    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "time": capture_time_str,   # ✅ ใช้เวลาที่ตั้งไว้
        "station_id": station_id,
        "province": province,
        "image_count": image_count,
        "bus": counts["bus"],
        "car": counts["car"],
        "motorcycle": counts["motorcycle"],
        "truck": counts["truck"],
        "van": counts["van"],
        "total_vehicles": total
    }


# === helper: รวมชื่อกล้องให้เป็น base เดียว (ลบ -IN / -OUT)
def base_camera_name(cam: str):
    return cam.replace("-IN", "").replace("-OUT", "")


# === รวมผลลัพธ์ของกล้อง base เดียวกัน
def merge_camera_results(results_all):
    merged = {}
    for res in results_all:
        base_cam = base_camera_name(res["station_id"])
        key = (res["date"], res["time"], base_cam, res["province"])
        if key not in merged:
            merged[key] = res.copy()
            merged[key]["station_id"] = base_cam
        else:
            merged[key]["image_count"] += res["image_count"]
            merged[key]["bus"] += res["bus"]
            merged[key]["car"] += res["car"]
            merged[key]["motorcycle"] += res["motorcycle"]
            merged[key]["truck"] += res["truck"]
            merged[key]["van"] += res["van"]
            merged[key]["total_vehicles"] += res["total_vehicles"]
    return list(merged.values())


# === แคปภาพตามช่วงเวลา
def main_capture(hour, minute):
    folder_name = datetime.now(bangkok_tz).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    ).strftime("%Y-%m-%d %H-%M-%S") + "G1"

    threads = []
    for province, cams in urls.items():
        for cam_key, url in cams.items():
            thread = threading.Thread(
                target=run_camera_capture,
                args=(province, cam_key, url, folder_name)
            )
            thread.start()
            threads.append(thread)

    for thread in threads:
        thread.join()

    # cleanup_duplicate_and_bad_images(folder_name)

    # === YOLO + insert DB ===
    capture_time_str = f"{hour:02d}:{minute:02d}:00" 

    results_all = []
    for province, cams in urls.items():
        for cam_key in cams:
            cam_dir = os.path.join(folder_name, province, cam_key)
            if os.path.isdir(cam_dir):
                base_cam = base_camera_name(cam_key)
                # ✅ ส่ง capture_time_str เข้าไป
                result = run_yolo_on_folder(cam_dir, base_cam, province, capture_time_str)
                results_all.append(result)


    # 🔹 รวมผล IN/OUT
    merged_results = merge_camera_results(results_all)

    for result in merged_results:
        insert_to_db(result)
        print(f"YOLO Done + DB: {result}")
    
    # === ลบโฟลเดอร์ที่สร้างขึ้นหลังประมวลผลเสร็จ ===
    try:
        shutil.rmtree(folder_name)
        print(f"🗑️ ลบโฟลเดอร์ {folder_name} เรียบร้อยแล้ว")
    except Exception as e:
        print(f"⚠️ ลบโฟลเดอร์ {folder_name} ไม่สำเร็จ: {e}")

    return merged_results


# === Scheduler
# if __name__ == "__main__":
#     for hour, minute in CAPTURE_TIMES:
#         now = datetime.now(bangkok_tz)
#         start_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

#         if now > start_time:
#             continue

#         wait_sec = (start_time - now).total_seconds()
#         time.sleep(wait_sec)

#         main_capture(hour, minute)

# === Scheduler
# if __name__ == "__main__":
#     for hour, minute in CAPTURE_TIMES:
#         now = datetime.now(bangkok_tz)
#         start_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

#         if now > start_time:
#             print(f"⏩ ข้ามรอบ {hour:02d}:{minute:02d} (เลยเวลาแล้ว)")
#             continue

#         wait_sec = (start_time - now).total_seconds()
#         print(f"⏳ กำลังรอเวลา {start_time.strftime('%Y-%m-%d %H:%M:%S')} "
#               f"(เหลือ {int(wait_sec)} วินาที)")
#         time.sleep(wait_sec)

#         print(f"\n🚀 เริ่มการแคปภาพ + YOLO สำหรับรอบ {hour:02d}:{minute:02d}\n")
#         main_capture(hour, minute)
#         print(f"\n✅ เสร็จสิ้นการแคปภาพ + YOLO สำหรับรอบ {hour:02d}:{minute:02d}\n")

def wait_until(target_time):
    """รอจนถึงเวลาที่กำหนด"""
    now = datetime.now(bangkok_tz)
    wait_seconds = (target_time - now).total_seconds()
    if wait_seconds > 0:
        print(f"⏳ รอเวลา {target_time.strftime('%H:%M:%S')} อีก {wait_seconds/60:.1f} นาที...")
        time.sleep(wait_seconds)


def daily_loop():
    """ลูปหลัก ทำงานในแต่ละวัน"""
    while True:
        today = datetime.now(bangkok_tz).date()

        for hour, minute in CAPTURE_TIMES:
            now = datetime.now(bangkok_tz)
            target_time = now.replace(hour=hour, minute=minute, second=0, microsecond=0)

            # ถ้าเวลานั้นผ่านมาแล้ววันนี้ → ข้าม
            if now > target_time:
                print(f"⏩ ข้ามรอบ {hour:02d}:{minute:02d} (เลยเวลาแล้ว)")
                continue

            # รอจนถึงเวลา
            wait_until(target_time)

            # เริ่มทำงาน
            print(f"🚀 เริ่มเก็บข้อมูลรอบ {hour:02d}:{minute:02d}")
            try:
                main_capture(hour, minute)
                print(f"✅ เสร็จสิ้นรอบ {hour:02d}:{minute:02d}")
            except Exception as e:
                print(f"❌ เกิดข้อผิดพลาดในรอบ {hour:02d}:{minute:02d} — {e}")

        # ✅ เมื่อจบวัน ให้รอวันใหม่ก่อนเริ่มรอบใหม่
        print("🌙 จบวัน รอวันใหม่...")
        while datetime.now(bangkok_tz).date() == today:
            time.sleep(60)  # เช็กทุก 1 นาที


if __name__ == "__main__":
    print("🚀 เริ่มต้นระบบเก็บข้อมูล (Cap_YOLO_DB3)")
    daily_loop()
