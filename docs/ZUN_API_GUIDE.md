# ZUN API — руководство для пользователя

Преподавательский анализ «знать / уметь / навык» (ЗУН): просмотр, поиск и редактирование ЗУН по направлениям обучения, связь с рыночными навыками и покрытием.

Машинная спецификация (OpenAPI 3.0.3): `docs/zun_api.json`.

---

## 1. Адреса API

| Среда | Базовый URL | Примечание |
|---|---|---|
| Прод (compareserver) | `https://compareserver.tailb909df.ts.net/api/teacher/zun/...` | Tailscale Funnel, nginx → backend :8000 |
| Публичный (laconiait.ru) | `https://laconiait.ru/api.php?target=/teacher/zun/...` | PHP-прокси на тот же backend |

---

## 2. Авторизация

1. Получить токен: `POST /api/auth/login` с телом `{"email": "...", "password": "..."}`.
2. Ответ: `{"token": "...", "role": "...", "name": "..."}`.
3. Передавать токен в каждый запрос заголовком: `Authorization: Bearer <token>`.

### Учётные записи (users.json)

| Email | Пароль | Роль |
|---|---|---|
| `admin@compare-competencies.local` | `admin` | admin |
| `teacher@compare-competencies.local` | `teacher123` | teacher |
| `student@compare-competencies.local` | `student` | student (доступа к ZUN нет) |

Роли с доступом к `/teacher/zun/*`: **admin, teacher, rop**.

### Пример (PowerShell)

```powershell
$r = Invoke-RestMethod -Uri "http://localhost:8000/api/auth/login" `
  -Method Post -ContentType "application/json" `
  -Body '{"email":"admin@compare-competencies.local","password":"admin"}'
$token = $r.token
$h = @{ Authorization = "Bearer $token" }
```

---

## 3. Эндпоинты

| Метод | Путь | Назначение |
|---|---|---|
| GET | `/teacher/zun/directions` | Направления + наличие ЗУН |
| GET | `/teacher/zun/my-directions` | Направления текущего РОП |
| GET | `/teacher/zun/stats?dir_code=09.03.02` | Сводка: дисциплины/компетенции/индикаторы/ЗУН/покрытие |
| GET | `/teacher/zun/disciplines?dir_code=` | Список дисциплин со счётчиками ЗУН |
| GET | `/teacher/zun/disciplines/{id}` | Дерево: компетенция → индикаторы → ЗУН |
| GET | `/teacher/zun/disciplines/name/{имя}` | Дерево по имени дисциплины |
| GET | `/teacher/zun/search?q=&ksa_type=` | Поиск ЗУН по подстроке (q ≥ 2 символа) |
| GET | `/teacher/zun/search/semantic?q=` | Семантический поиск по тексту ЗУН (эмбеддинги) |
| GET | `/teacher/zun/filter?category=&code_prefix=` | Фильтр компетенций (УК/ОПК/ПК/ППК/ИП/ВПК) |
| GET | `/teacher/zun/competencies/{id}/skills` | Рыночные навыки компетенции |
| GET | `/teacher/zun/competencies/{id}/coverage` | Покрытие компетенции |
| GET | `/teacher/zun/analyze/results?dir_code=` | Результаты анализа |
| POST | `/teacher/zun/competencies/{id}/entries` | Добавить ЗУН → 201 |
| PATCH | `/teacher/zun/entries/{ksa_id}` | Изменить текст ЗУН |
| DELETE | `/teacher/zun/entries/{ksa_id}` | Удалить ЗУН |
| POST | `/teacher/zun/analyze?dir_code=` | Запустить анализ (фон) |
| GET | `/teacher/zun/analyze/status/{run_id}` | Статус запуска анализа |
| POST | `/teacher/zun/import/{dir_code}` | Импорт ЗУН из KRM JSON (идемпотентно) |

`dir_code` по умолчанию — `09.03.02`. Пагинация: `limit` (до 500, дефолт 100) + `offset`.

---

## 4. Примеры запросов

Базовый URL опущен — подставляй адрес из раздела 1.

### Сводка по направлению

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/stats?dir_code=09.03.02" -Headers $h
```

Ответ:
```json
{
  "dir_code": "09.03.02",
  "total_disciplines": 60,
  "total_competencies": 189,
  "total_indicators": 394,
  "ksa_counts": { "knowledge": 1878, "abilities": 1658, "skills": 1518, "total": 5054 },
  "linked_skills": 1073,
  "coverage": { "avg_coverage_ratio": 0.7434, "covered_disciplines": 28 }
}
```

### Поиск по подстроке

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/search?q=python&dir_code=09.03.02" -Headers $h
```

### Семантический поиск

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/search/semantic?q=машинное%20обучение&limit=10" -Headers $h
```

Ответ: `{"mode":"semantic","matches":[{"original_text":"...","similarity":0.89,...}]}`. Если эмбеддинги недоступны — `"mode":"fallback_substring"`.

### Добавить ЗУН

```powershell
$body = @{ ksa_type = "knowledge"; text = "Знает принципы машинного обучения" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/competencies/{competency_id}/entries" `
  -Method Post -Headers $h -ContentType "application/json" -Body $body
```

→ `201 {"ksa_id": "...", "ksa_type": "knowledge", "text": "..."}`. Повторный POST того же текста → `409`.

### Изменить / удалить ЗУН

```powershell
# изменить
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/entries/{ksa_id}" `
  -Method Patch -Headers $h -ContentType "application/json" -Body '{"text":"новая формулировка"}'

# удалить
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/entries/{ksa_id}" `
  -Method Delete -Headers $h
```

### Импорт из KRM JSON (проверка без записи)

```powershell
Invoke-RestMethod -Uri "http://localhost:8000/api/teacher/zun/import/09.03.02?dry_run=true" `
  -Method Post -Headers $h
```

Ответ: `{"dir_code":"09.03.02","inserted":905,"skipped_dups":139,"skipped_gap":0,"disciplines_processed":26}`.

---

## 5. Коды ошибок

| Код | Причина |
|---|---|
| 400 | Неверный параметр: q < 2 символов, плохой category/match_type, не-UUID id, неверный dir_code |
| 401 | Нет или невалидный токен |
| 403 | Роль не подходит (нужна admin/teacher/rop) |
| 404 | Направление/дисциплина/компетенция/запись/результат не найдены |
| 409 | Дубль ЗУН при добавлении |

---

## 6. Типичный сценарий преподавателя

1. Логин → токен.
2. `GET /teacher/zun/directions` → выбрать направление.
3. `GET /teacher/zun/stats?dir_code=...` → общая картина.
4. `GET /teacher/zun/disciplines?dir_code=...` → список дисциплин.
5. `GET /teacher/zun/disciplines/{id}` → ЗУН конкретной дисциплины (компетенции и индикаторы).
6. `POST /teacher/zun/competencies/{id}/entries` → добавить недостающий ЗУН.
7. `PATCH /teacher/zun/entries/{ksa_id}` → поправить формулировку.
8. `GET /teacher/zun/search?q=...` → найти, где встречается термин.
