# Multi-stage build: trading_system_pro
# TA-Lib built from source (not in Bookworm main repos)
# Stage 1: build TA-Lib C library + Python deps
FROM python:3.11-slim-bookworm AS builder
WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Build and install TA-Lib C library from source
ARG TALIB_VERSION=0.4.0
RUN wget -q -L "https://sourceforge.net/projects/ta-lib/files/ta-lib/${TALIB_VERSION}/ta-lib-${TALIB_VERSION}-src.tar.gz/download" -O ta-lib.tar.gz \
    && tar xzf ta-lib.tar.gz \
    && cd ta-lib \
    && sed -i 's|0.00000001|0.0000001|g' src/ta_func/ta_utility.h \
    && ./configure --prefix=/usr/local \
    && make -j1 \
    && make install \
    && cd .. && rm -rf ta-lib ta-lib.tar.gz

COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: runtime
FROM python:3.11-slim-bookworm AS runtime
WORKDIR /app

# Copy TA-Lib shared library from builder (no apt package in bookworm)
COPY --from=builder /usr/local/lib/libta_lib.* /usr/local/lib/
RUN echo /usr/local/lib > /etc/ld.so.conf.d/ta-lib.conf && ldconfig

# Non-root user
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Copy installed packages from builder
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH
ENV PYTHONPATH=/app:/app/src
ENV TZ=UTC

# Copy application
COPY --chown=appuser:appuser . .

# Writable dirs + full /app ownership so appuser can create dirs at runtime
USER root
RUN mkdir -p /app/data/models /app/data/datasets /app/data/backups /app/logs /app/ml_models \
    /app/paper_trades /app/portfolio /app/simulation_logs \
    /app/signals /app/deepseek_logs /app/real_orders \
    && chown -R appuser:appuser /app
USER appuser

# Healthcheck: import from src (post-refactor)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from src.core.config import settings; print('ok')" || exit 1

EXPOSE 8501 8502
# Padrão: modo monitor (sobe tudo com docker compose)
CMD ["python", "main.py", "--mode", "monitor"]
