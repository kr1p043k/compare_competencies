"""
Дополнительный эндпоинт для API - добавьте этот код в ваш FastAPI файл
"""

# Добавьте в начало файла с импортами:
import json
from pathlib import Path

# Добавьте глобальную переменную после других:
basic_vacancies = []  # Список всех вакансий

# Добавьте в функцию startup() после строки 80:
# global basic_vacancies  # добавьте эту строку в список global переменных

# Добавьте эти эндпоинты в конец файла (перед if __name__ == "__main__":):

@app.get("/api/vacancies")
async def get_vacancies(
    limit: int = Query(50, ge=1, le=500, description="Количество вакансий"),
    offset: int = Query(0, ge=0, description="Смещение для пагинации"),
    experience: str | None = Query(None, description="Фильтр по опыту: junior, middle, senior"),
    search: str | None = Query(None, description="Поиск по названию")
):
    """Получить список вакансий с фильтрами"""
    filtered = basic_vacancies.copy()

    # Фильтр по опыту
    if experience:
        exp_lower = experience.lower()
        filtered = [
            v for v in filtered
            if (isinstance(v.get("experience"), dict) and
                exp_lower in v["experience"].get("id", "").lower()) or
               (isinstance(v.get("experience"), str) and
                exp_lower in v["experience"].lower()) or
               exp_lower in v.get("name", "").lower()
        ]

    # Фильтр по поиску
    if search:
        search_lower = search.lower()
        filtered = [
            v for v in filtered
            if search_lower in v.get("name", "").lower() or
               search_lower in v.get("description", "").lower()
        ]

    total = len(filtered)
    items = filtered[offset:offset + limit]

    # Форматируем данные
    formatted_items = []
    for vac in items:
        # Извлекаем навыки
        skills = []
        if "extracted_skills" in vac:
            skills = vac["extracted_skills"][:10]  # Первые 10 навыков

        # Определяем уровень
        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id or "no_experience" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"

        # Зарплата
        salary_from = None
        salary_to = None
        salary_currency = "RUR"
        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            salary_from = sal.get("from")
            salary_to = sal.get("to")
            salary_currency = sal.get("currency", "RUR")

        # Работодатель
        employer_name = "Не указано"
        employer_logo = None
        if "employer" in vac and vac["employer"]:
            emp = vac["employer"]
            employer_name = emp.get("name", "Не указано")
            if "logo_urls" in emp and emp["logo_urls"]:
                employer_logo = emp["logo_urls"].get("240") or emp["logo_urls"].get("90")

        formatted_items.append({
            "id": vac.get("id"),
            "name": vac.get("name", "Без названия"),
            "experience": exp,
            "salary_from": salary_from,
            "salary_to": salary_to,
            "salary_currency": salary_currency,
            "employer_name": employer_name,
            "employer_logo": employer_logo,
            "area": vac.get("area", {}).get("name", "Не указано") if isinstance(vac.get("area"), dict) else "Не указано",
            "published_at": vac.get("published_at"),
            "alternate_url": vac.get("alternate_url"),
            "skills": skills,
            "snippet": vac.get("snippet", {})
        })

    return {
        "items": formatted_items,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total
    }


@app.get("/api/vacancies/{vacancy_id}")
async def get_vacancy_detail(vacancy_id: str):
    """Получить детальную информацию о вакансии"""
    for vac in basic_vacancies:
        if vac.get("id") == vacancy_id:
            # Извлекаем все навыки
            skills = []
            if "extracted_skills" in vac:
                skills = vac["extracted_skills"]

            return {
                "id": vac.get("id"),
                "name": vac.get("name"),
                "description": vac.get("description", ""),
                "experience": vac.get("experience"),
                "salary": vac.get("salary"),
                "employer": vac.get("employer"),
                "area": vac.get("area"),
                "published_at": vac.get("published_at"),
                "alternate_url": vac.get("alternate_url"),
                "skills": skills,
                "schedule": vac.get("schedule"),
                "employment": vac.get("employment"),
                "key_skills": vac.get("key_skills", []),
                "snippet": vac.get("snippet")
            }

    raise HTTPException(status_code=404, detail="Вакансия не найдена")


@app.get("/api/vacancies/stats/summary")
async def get_vacancies_stats():
    """Получить статистику по вакансиям"""
    total = len(basic_vacancies)

    # Подсчет по уровням
    junior = 0
    middle = 0
    senior = 0

    # Зарплаты
    salaries = []

    for vac in basic_vacancies:
        # Уровень
        exp = "middle"
        if "experience" in vac:
            exp_obj = vac["experience"]
            if isinstance(exp_obj, dict):
                exp_id = exp_obj.get("id", "").lower()
                if "junior" in exp_id or "less1" in exp_id:
                    exp = "junior"
                elif "senior" in exp_id or "morethan10" in exp_id:
                    exp = "senior"

        if exp == "junior":
            junior += 1
        elif exp == "senior":
            senior += 1
        else:
            middle += 1

        # Зарплаты
        if "salary" in vac and vac["salary"]:
            sal = vac["salary"]
            if sal.get("from"):
                salaries.append(sal["from"])
            if sal.get("to"):
                salaries.append(sal["to"])

    avg_salary = sum(salaries) / len(salaries) if salaries else 0

    return {
        "total": total,
        "by_experience": {
            "junior": junior,
            "middle": middle,
            "senior": senior
        },
        "salary": {
            "average": round(avg_salary, 0),
            "min": min(salaries) if salaries else 0,
            "max": max(salaries) if salaries else 0,
            "count": len(salaries)
        }
    }
