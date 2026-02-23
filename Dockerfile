FROM python:3.11-slim

WORKDIR /app

# Install system dependencies needed by scripts
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    curl \
    lsof \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy project files
COPY . /app

# Make scripts executable
RUN chmod +x init.sh startup.sh clean.sh scripts/*.py 2>/dev/null || true

# Install dependencies system-wide (Docker doesn't need venv)
RUN uv pip install --system -r config/requirements.txt

# Create necessary directories
RUN mkdir -p logs data/faiss_index

# Create .venv structure for startup.sh compatibility
# startup.sh expects .venv/bin/python to exist and uses source .venv/bin/activate
# We create a minimal venv structure that uv will accept
RUN mkdir -p .venv/bin && \
    ln -sf /usr/local/bin/python3 .venv/bin/python && \
    ln -sf /usr/local/bin/python3 .venv/bin/python3 && \
    echo 'home = /usr/local\ninclude-system-site-packages = true\nversion = 3.11.14' > .venv/pyvenv.cfg && \
    echo '#!/bin/bash\n# Minimal activate script for Docker compatibility\nexport VIRTUAL_ENV=/app/.venv\nexport PATH="/app/.venv/bin:$PATH"\n' > .venv/bin/activate && \
    chmod +x .venv/bin/activate

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose ports (8000 for FastAPI, 8501 for Streamlit)
EXPOSE 8000 8501

# Copy and use entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# Use entrypoint script that handles init.sh and startup.sh
# Makes data sync optional so container can start even if GCS is unreachable
ENTRYPOINT ["/docker-entrypoint.sh"]

