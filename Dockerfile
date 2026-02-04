FROM python:3.11-slim  
  
WORKDIR /app  
  
# Install system dependencies  
RUN apt-get update && apt-get install -y \  
    && rm -rf /var/lib/apt/lists/*  
  
# Copy requirements first for better caching  
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Copy application code  
COPY . .  
  
# Expose port  
EXPOSE 8800  
  
# Set environment variables  
ENV PARLANT_HOST=0.0.0.0  
ENV PARLANT_PORT=8800  
  
# Run the application  
CMD ["python", "main.py"]
