/**
 * Проверка hash-роутера: deep link, смена вкладки меняет адрес,
 * неизвестный адрес откатывается, кнопка «назад» работает.
 */
const path = require("path");
const { createRequire } = require("module");

const frontendRequire = createRequire(path.resolve(__dirname, "..", "..", "frontend", "package.json"));
const WebSocket = frontendRequire("ws");

const CDP = "http://127.0.0.1:9222";
const APP = "http://localhost:3000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const targets = await (await fetch(`${CDP}/json/list`)).json();
  const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false, maxPayload: 256 * 1024 * 1024 });
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });

  let id = 0;
  const pending = new Map();
  ws.on("message", (d) => {
    const m = JSON.parse(d.toString());
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  });
  const send = (method, params = {}, ms = 20000) => new Promise((res, rej) => {
    const i = ++id;
    const t = setTimeout(() => rej(new Error(`timeout ${method}`)), ms);
    pending.set(i, (r) => { clearTimeout(t); res(r); });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  const evaluate = async (expr, ms = 20000) => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }, ms);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  };

  await send("Page.enable");
  await send("Runtime.enable");

  // сессия через API, как в visual-check
  const login = await fetch("http://localhost:8000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: "admin@compare-competencies.local", password: "admin" }),
  });
  const session = await login.json();

  await send("Page.navigate", { url: APP });
  await sleep(2500);
  await evaluate(`localStorage.setItem("auth", ${JSON.stringify(JSON.stringify(session))})`);

  const results = [];

  // 1. deep link на конкретную вкладку
  await send("Page.navigate", { url: `${APP}/#/teacher` });
  await sleep(5000);
  results.push({
    step: "deep link #/teacher",
    hash: await evaluate("location.hash"),
    teacherAnalisisVisible: await evaluate(`document.body.innerText.includes("Промышленный") || document.body.innerText.includes("дисциплин") || document.body.innerText.includes("Анализ РПД")`),
    heading: await evaluate(`document.querySelector("h1")?.innerText`),
  });

  // 2. клик по вкладке меняет адрес
  await evaluate(`(async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
    const realClick = (el) => {
      const o = { bubbles: true, cancelable: true, button: 0, pointerId: 1, isPrimary: true };
      el.dispatchEvent(new PointerEvent('pointerdown', o));
      el.dispatchEvent(new PointerEvent('pointerup', o));
      el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    };
    for (const t of Array.from(document.querySelectorAll('button[aria-haspopup],[data-state]'))) {
      realClick(t); await sleep(350);
      const item = Array.from(document.querySelectorAll('[role="menuitem"]')).find((el) => norm(el.innerText) === 'Помощь');
      if (item) { realClick(item); return 'ok'; }
      document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); await sleep(120);
    }
    return 'not-found';
  })()`);
  await sleep(2500);
  results.push({ step: "клик «Помощь»", hash: await evaluate("location.hash"), navCalls: await evaluate("history.length") });

  // 3. неизвестный hash -> fallback
  await send("Page.navigate", { url: `${APP}/#/нетакой-вкладки` });
  await sleep(4000);
  results.push({ step: "неизвестный hash", hash: await evaluate("location.hash") });

  // 4. назад
  await evaluate("history.back()");
  await sleep(2500);
  results.push({ step: "history.back()", hash: await evaluate("location.hash") });

  console.log(JSON.stringify(results, null, 2));
  ws.close();
  process.exit(0);
})().catch((e) => { console.error("FATAL", e.message); process.exit(1); });
