# 1. Gunakan base image Node.js versi Linux
FROM node:18-slim

# 2. Instal library yang dibutuhkan
RUN apt-get update && apt-get install -y \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# 3. Tentukan direktori kerja
WORKDIR /app

# 4. Salin file package.json dan package-lock.json
COPY package*.json ./

# 5. Install dependensi secara eksplisit
RUN npm install --build-from-source

# 6. Salin seluruh project ke container
COPY . .

# 7. Verifikasi file di dalam container
RUN ls -R /app

# 8. Expose port untuk container
EXPOSE 8080

# 9. Jalankan aplikasi
CMD ["node", "app.js"]