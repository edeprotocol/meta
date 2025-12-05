# Synthetic Field Layer - Docker Image
FROM python:3.11-slim

LABEL maintainer="EDE Protocol <contact@edeprotocol.com>"
LABEL description="Synthetic Field Layer - The Economic Operating System for AGI/ASI"

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create app directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install .

# Copy application code
COPY sfl/ ./sfl/
COPY README.md .

# Create non-root user
RUN useradd -m -u 1000 sfl && \
    chown -R sfl:sfl /app

USER sfl

# Create data directory
RUN mkdir -p /app/field_memory

# Expose port
EXPOSE 8420

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8420/health').raise_for_status()"

# Default command
CMD ["python", "-m", "sfl.api.server"]
