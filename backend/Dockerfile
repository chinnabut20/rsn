# ใช้ Python slim image (เหมาะกับ TensorFlow 2.20 + NumPy 2.x)
FROM python:3.10-slim

# กำหนด working directory
WORKDIR /app

# ติดตั้ง system dependencies ที่จำเป็น
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    gfortran \
    libhdf5-dev \
    libatlas-base-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt เข้ามา
COPY requirements.txt .

# ติดตั้ง torch/torchvision (CPU version)
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# 🔥 ติดตั้ง dependencies ทั้งหมดตาม requirements.txt (เวอร์ชันตรง)
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกซอร์สโค้ดทั้งหมด
COPY . .

# เปิด port FastAPI
EXPOSE 8000

# รัน FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
