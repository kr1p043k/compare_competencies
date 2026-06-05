$chatId = "1435041436"
$token = $env:TG_BOT_TOKEN
$msg = @"
✅ <b>compare_competencies — работа завершена</b>

<b>Сделано:</b>
• Проверены Result[T,E] утверждения по 5 файлам
• Исправлено: artifacts.py, skill_parser.py, main.py
• ML Architecture Abstraction Layer для rerankers: BaseReranker, HybridReranker, RerankerBuilder
• Скачаны все PDF (123 шт): РПД, РПП, ГИА, Аннотации, ОПОП, Приложения
• RPDLoader запущен: <b>43 дисциплины, 1174 навыка</b>
• 180/181 тестов проходят

<b>Остаётся:</b>
• Очистка извлечённых навыков (много шума от OCR)
• DisciplineHealth + CurriculumGraph анализаторы
• Обновление профилей 09.03.02_база/профиль/эксперт
• Кросс-фильтр в CurriculumRecommender
• Тесты на новые модули
"@

$body = @{
    chat_id = $chatId
    text = $msg
    parse_mode = "HTML"
} | ConvertTo-Json

Invoke-RestMethod -Uri "https://api.telegram.org/bot$token/sendMessage" -Method Post -Body $body -ContentType "application/json"
