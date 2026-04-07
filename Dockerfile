FROM python:3.11-slim

# ── Metadata ──────────────────────────────────────────────────────────────────
LABEL maintainer="Scalar Hackathon Team"
LABEL description="Startup Decision Simulator — OpenEnv Compatible Environment"
LABEL version="1.0.0"

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (layer cached separately) ─────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY environment/ ./environment/
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# ── Environment variables (can be overridden at runtime) ──────────────────────
ENV API_BASE_URL="https://api.openai.com/v1"
ENV MODEL_NAME="gpt-4o-mini"
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ── Non-root user for security ────────────────────────────────────────────────
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# ── Healthcheck ───────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from environment import StartupEnv; env = StartupEnv(); env.reset(); print('OK')" || exit 1

# ── Default command ───────────────────────────────────────────────────────────
CMD ["python", "inference.py"]
