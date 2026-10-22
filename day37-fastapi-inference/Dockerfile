
# Base Python runtime
FROM python:3.11-slim

# Python runtime configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Run as an unprivileged user
RUN useradd --create-home --uid 10001 appuser
USER appuser

# Application port
EXPOSE 8000

# Start inference server
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000"]
