FROM python:3.11-slim

WORKDIR /app

# Устанавливаем системные зависимости
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
RUN pip install --no-cache-dir -r requirements.txt --timeout=1000 --default-timeout=1000

# Загружаем модель во время сборки напрямую (без отдельного файла)
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); print('✅ Модель загружена')"

# Копируем исходный код
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
    /app/uploads \
    /root/.cache/huggingface

# Настройки окружения для Hugging Face
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/root/.cache/huggingface \
    HF_HUB_CACHE=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface \
    HF_HUB_OFFLINE=0 \
    TRANSFORMERS_OFFLINE=0 \
    HF_ENDPOINT=https://hf-mirror.com

# Healthcheck для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000"]