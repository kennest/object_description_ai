# syntax=docker/dockerfile:1.4
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system dependencies for OpenCV and git for pip packages
RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip for faster dependency resolution
RUN pip install --upgrade pip

# Install heavy dependencies first (better layer caching)
# These rarely change, so they'll be cached most of the time
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy and install remaining requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# Create Ultralytics config directory
RUN mkdir -p /tmp/Ultralytics && chmod 777 /tmp/Ultralytics

# Copy application code
COPY . .

# Expose port
EXPOSE 7100

# Run the application
CMD ["uvicorn", "main_llava:app", "--host", "0.0.0.0", "--port", "7100"]
