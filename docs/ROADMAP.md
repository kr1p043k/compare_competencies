# Roadmap улучшений

## 🔴 Критические (ближайшие)

| # | Улучшение | Усилие |
|---|-----------|--------|
| 28 | Strict mypy coverage | 1-2 ч |
| 29 | Pydantic response validation на всех эндпоинтах API | 2-3 ч |
| 47 | Taxonomy consistency validator | 1-2 ч |
| 60 | Formal architecture docs + ADR | 2-3 ч |
| 26 | Typed pipeline context object (вместо dict) | 3 ч |
| 14 | Unified normalization pipeline | 3-4 ч |
| 2 | Result[T, E] / Either pattern для ошибок | 5-6 ч |
| 3 | Unified Artifact Registry | 4 ч |
| 4 | Typed Feature Contracts | 3-4 ч |
| 23 | Hybrid retriever refactor (pluggable BM25/semantic/graph) | 6-7 ч |
| 16 | Explainability engine v2 | 4 ч |
| 11 | Domain Event Bus | 5 ч |
| 5 | Feature Registry + auto-discovery | 5-6 ч |
| 1 | Pipeline Orchestrator (DAG + retries) | 7-10 ч |
| 6 | Full profession ontology graph | 8 ч |

## 🟡 Важные

| # | Улучшение | Усилие |
|---|-----------|--------|
| 7 | Skill evolution timeline engine | 4-5 ч |
| 8 | Competency graph embeddings | 6-8 ч |
| 9 | Semantic career transition engine | 6-7 ч |
| 10 | Incremental taxonomy learning | 5-6 ч |
| 12-25 | Компараторы, эмбеддинги, confidence, temporal, графы | 3-9 ч |
| 30-46 | Exceptions, batch, ANN, benchmark, fuzz, ranking... | 2-7 ч |
| 48-57 | Memory, gap scoring, transferability, anomaly detection | 2-7 ч |

## 🟢 Перспективные

| # | Улучшение | Усилие |
|---|-----------|--------|
| 51 | Career trajectory simulator | 7-9 ч |
| 58 | Career-path recommendation engine | 7-10 ч |
| 59 | Meta-ranking ensemble | 6-8 ч |
| 50 | Research-grade evaluation reports | 4-5 ч |

## Быстрый старт (этапность)

1. **Этап 1 (1-2 ч):** `#28` + `#47` + `#60` — mypy, валидация таксономии, ARCHITECTURE.md
2. **Этап 2 (3-4 ч):** `#26` + `#29` + `#14` — PipelineContext, Pydantic API, нормализация
3. **Этап 3 (4-6 ч):** `#2` + `#3` + `#4` — Result тип, Artifact Registry, Feature Contracts
4. **Этап 4:** `#23`, `#1`, `#6`, `#11` — крупные архитектурные изменения
