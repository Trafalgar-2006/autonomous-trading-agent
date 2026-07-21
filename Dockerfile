# Trading Agent — reproducible runtime image.
FROM python:3.11-slim

# Non-root user (the agent never needs root).
RUN useradd --create-home --shell /bin/bash agent

WORKDIR /app

# Install deps first so code changes don't bust the layer cache.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -e .

COPY config ./config

# Runtime state (SQLite DB, model, cache, heartbeat) lives here — mount a volume.
RUN mkdir -p /app/data && chown -R agent:agent /app
USER agent

ENV PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8

# Liveness: the agent touches data/heartbeat.txt each cycle.
HEALTHCHECK --interval=5m --timeout=10s --start-period=2m --retries=3 \
    CMD python -m src.ops.healthcheck || exit 1

CMD ["python", "-m", "src.main", "run"]
