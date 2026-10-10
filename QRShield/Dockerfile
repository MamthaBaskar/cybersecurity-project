# Dockerfile for QRShield on Render
FROM python:3.11-slim

# Install required Linux system libraries for pyzbar and OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project code, models, and frontend
COPY . .

# Render injects the PORT environment variable
ENV PORT=8000
EXPOSE 8000

# Run uvicorn server from the backend module
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

