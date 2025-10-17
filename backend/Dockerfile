FROM python:3.11-slim

WORKDIR /app

# ติดตั้ง dependencies ระบบ
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    gcc \
    gfortran \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ติดตั้ง torch CPU
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu

# ติดตั้งตาม requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
