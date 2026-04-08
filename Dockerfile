FROM python:3.11-slim

LABEL maintainer="Scalar Hackathon Team"
LABEL description="Startup Decision Simulator — OpenEnv Compatible Environment"
LABEL version="1.0.0"

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
COPY pyproject.toml .
COPY uv.lock .
RUN pip install --no-cache-dir -r requirements.txt

COPY environment/ ./environment/
COPY inference.py .
COPY app.py .
COPY openenv.yaml .
COPY README.md .

ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.3-70b-versatile"
ENV GROQ_API_KEY=""
ENV HF_TOKEN=""
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# HF Spaces runs as user 1000
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["python", "app.py"]
