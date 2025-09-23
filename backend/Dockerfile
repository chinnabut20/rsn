# ใช้ Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# ติดตั้ง system dependencies (สำหรับ OpenCV)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# คัดลอก requirements.txt
COPY requirements.txt .

# ติดตั้ง torch/torchvision
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# Fix numpy ให้เข้ากับ torch
RUN pip install --no-cache-dir "numpy<2"

# ติดตั้ง dependencies อื่น ๆ
RUN pip install --no-cache-dir -r requirements.txt

# คัดลอกโค้ดที่เหลือ
COPY . .

# เปิด port FastAPI
EXPOSE 8000

# Run FastAPI (production mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
