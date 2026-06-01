FROM python:3.11-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .

# Установка с дополнительными опциями для экономии места
RUN pip install --no-cache-dir \
    --no-deps \
    -r requirements.txt 2>/dev/null || \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ ./src/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY main.py .
COPY pyproject.toml .

# Создание директорий
RUN mkdir -p /app/data/cache /app/data/result /app/data/processed /app/data/history /app/logs

EXPOSE 8000

CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000"]