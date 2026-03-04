# Multi-stage build: trading_system_pro
# Stage 1: build dependencies
FROM python:3.11-slim AS builder
WORKDIR /build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libta-lib0-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libta-lib0 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# Copy application
COPY --chown=appuser:appuser . .

# Healthcheck (assume main.py can be run with --help)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import config; print('ok')" || exit 1

EXPOSE 8501
CMD ["python", "main.py", "--symbol", "BTCUSDT", "--mode", "single"]
