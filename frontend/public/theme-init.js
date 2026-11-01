/* Тема до рендера: без вспышки светлой темы при dark.
   Вынесено из index.html в файл, чтобы отдавать строгий CSP script-src 'self'
   (инлайн-скрипт требует 'unsafe-inline' или хеша). */
try {
  if (localStorage.getItem("cc-theme") === "dark") {
    document.documentElement.classList.add("dark");
    document.documentElement.style.colorScheme = "dark";
  }
} catch (e) {
  /* localStorage может быть недоступен (приватный режим) - молча продолжаем */
}
