FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for OpenCV
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create Ultralytics config directory
RUN mkdir -p /tmp/Ultralytics && chmod 777 /tmp/Ultralytics

# Copy application code
COPY . .

# Expose port
EXPOSE 7100

# Run the application
CMD ["uvicorn", "main_llava:app", "--host", "0.0.0.0", "--port", "7100"]
