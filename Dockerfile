FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Сначала устанавливаем numpy фиксированной версии
RUN pip install --no-cache-dir numpy==1.24.3

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY main.py .
COPY pyproject.toml .

RUN mkdir -p /app/data/cache \
    /app/data/result \
    /app/data/processed \
    /app/data/history \
    /app/logs \
    /app/uploads

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000"]