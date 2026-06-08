FROM python:3.12-slim

WORKDIR /app

# System deps for tree-sitter and sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caches until requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# HuggingFace model cache lives outside the container image.
# Mount a volume at /root/.cache/huggingface to persist across restarts.
ENV HF_HOME=/root/.cache/huggingface

# Expose the API port
EXPOSE 8002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8002"]
