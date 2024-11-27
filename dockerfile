# 1. Gunakan base image Node.js
FROM node:18.20.5

# 2. Tentukan direktori kerja di dalam container
WORKDIR /app

# 3. Salin file package.json dan package-lock.json
COPY package*.json ./

# 4. Install dependensi aplikasi
RUN npm install
RUN npm rebuild @tensorflow/tfjs-node
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev

# 5. Salin semua file dari proyek Anda ke dalam container
COPY . .

# 6. Expose port 8000
EXPOSE 8080

# 7. Tentukan perintah untuk menjalankan aplikasi
CMD ["node", "app.js"]