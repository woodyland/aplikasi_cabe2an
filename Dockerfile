FROM python:3.9-slim

WORKDIR /app

# Install system dependencies untuk OpenCV
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend files
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Create models directory
RUN mkdir -p models

EXPOSE 8000

CMD ["uvicorn", "simple_main:app", "--host", "0.0.0.0", "--port", "$PORT"]