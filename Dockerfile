FROM python:3.12-slim

WORKDIR /app

# Непривилегированный пользователь (BACKLOG 6.7.2)
RUN groupadd --system app && useradd --system --gid app --home-dir /app app

# Кэш HuggingFace внутри /app, чтобы он был доступен непривилегированному
# пользователю (раньше был /root/.cache, недоступный для non-root).
ENV HF_HOME=/app/.cache/huggingface \
    HF_HUB_CACHE=/app/.cache/huggingface/hub \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/hub \
    HF_ENDPOINT=https://hf-mirror.com
RUN mkdir -p /app/.cache/huggingface/hub

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    libpq-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем torch (CPU-only, без CUDA)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Копируем и устанавливаем остальные зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --timeout=1000 --default-timeout=1000

# Загружаем модель во время сборки напрямую (без отдельного файла)
RUN python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2'); print('✅ Модель загружена')"

# Копируем исходный код
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY main.py .
COPY pyproject.toml .
COPY alembic.ini .
COPY alembic/ ./alembic/

# Создаем необходимые директории
RUN mkdir -p /app/data/cache \
    /app/data/result \
    /app/data/processed \
    /app/data/history \
    /app/logs \
    /app/uploads \
    && chown -R app:app /app

# Настройки окружения для Hugging Face (пути заданы выше)
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

# Запуск от непривилегированного пользователя (6.7.2)
USER app

# Healthcheck для Docker
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Открываем порт
EXPOSE 8000

# Запуск приложения
CMD ["uvicorn", "src.api_pkg:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-keep-alive", "65", "--limit-concurrency", "200", "--backlog", "100"]