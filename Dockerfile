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

# Устанавливаем numpy фиксированной версии
RUN pip install --no-cache-dir numpy==1.24.3

# Копируем и устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходный код (без папки scripts, которой нет в проекте)
COPY src/ ./src/
COPY data/ ./data/
COPY main.py .
COPY pyproject.toml .

# Создаем необходимые директории
RUN mkdir -p /app/data/cache \
    /app/data/result \
    /app/data/processed \
    /app/data/history \
    /app/logs \
    /app/uploads

# Настройки окружения
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Healthcheck для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000"]