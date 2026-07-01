# Use a lightweight python base image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

# Set working directory
WORKDIR /app

# Install system dependencies required by OpenCV and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU first to keep image sizes smaller
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all application code
COPY . .

# Create screenshots and uploads folders
RUN mkdir -p screenshots uploads

# Expose the application port
EXPOSE 10000

# Start the Flask web app
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "web_app:app"]