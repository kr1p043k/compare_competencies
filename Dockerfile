FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY main.py .
COPY pyproject.toml .

RUN mkdir -p /app/data/cache /app/data/result /app/data/processed /app/data/history /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000"]