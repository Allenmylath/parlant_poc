# Use Python 3.10 slim image (Parlant requires Python 3.10)
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies with verbose output
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY models.py .
COPY tools.py .

# Expose port (Render will set PORT env variable)
EXPOSE 8000

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python", "main.py"]
