FROM python:3.12-slim

WORKDIR /app

# git: optional fetch_docs.py; curl: healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caches until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

EXPOSE 8002

CMD ["uvicorn", "livedocs.query.app:app", "--host", "0.0.0.0", "--port", "8002"]
