FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set PYTHONPATH environment variable to ensure package resolution
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Start server
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
