# ใช้ Node.js เป็น base image
FROM node:latest

# กำหนด working directory ใน container
WORKDIR /app

# คัดลอกไฟล์ package.json และ package-lock.json ไปยัง working directory
COPY package*.json ./

# ติดตั้ง dependencies ที่ระบุใน package.json
RUN npm install

# คัดลอกโค้ดทั้งหมดจากเครื่อง local ไปยัง container
COPY . .

# ติดตั้ง nodemon สำหรับโหมดการพัฒนา
RUN npm install -g nodemon

# เปิดพอร์ตที่แอปพลิเคชันจะใช้งาน (ปกติจะเป็น 3000 สำหรับ React)
EXPOSE 3000

# คำสั่งสำหรับรันแอปพลิเคชันโดยใช้ nodemon
CMD ["nodemon", "src/index.js"]
