# syntax=docker/dockerfile:1

# Build stage for dependencies
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml .

# Create requirements.txt from pyproject.toml and install dependencies
RUN pip install --no-cache-dir pip-tools && \
    pip-compile --output-file=requirements.txt pyproject.toml && \
    pip wheel --no-cache-dir --wheel-dir=/app/wheels -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    APP_HOME=/app \
    APP_USER=appuser

WORKDIR $APP_HOME

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd --gid 1000 $APP_USER \
    && useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home $APP_USER

# Copy wheels from builder and install
COPY --from=builder /app/wheels /wheels
COPY --from=builder /app/requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy application code
COPY --chown=$APP_USER:$APP_USER src/ src/
COPY --chown=$APP_USER:$APP_USER config/ config/

# Create directories for logs and cache with proper permissions
RUN mkdir -p logs .llm_cache \
    && chown -R $APP_USER:$APP_USER logs .llm_cache

# Switch to non-root user
USER $APP_USER

# Expose Streamlit default port
EXPOSE 8501

# Health check - Streamlit exposes /_stcore/health
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Configure Streamlit
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Run Streamlit app
ENTRYPOINT ["python", "-m", "streamlit", "run"]
CMD ["src/murban_copilot/interface/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
