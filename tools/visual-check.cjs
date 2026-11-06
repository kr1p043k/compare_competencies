/**
 * Минимальный CDP-драйвер на голом ws (без puppeteer).
 * Обходит вкладки приложения и собирает ошибки консоли, метрики раскладки
 * и скриншоты. У каждого вызова есть таймаут: если приложение подвисло,
 * прогон не виснет целиком, а помечает вкладку как timeout.
 */
const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");

const frontendRequire = createRequire(path.resolve(__dirname, "..", "frontend", "package.json"));
const WebSocket = frontendRequire("ws");

const CDP = "http://127.0.0.1:9222";
const APP = "http://localhost:3000";
const OUT = path.resolve(__dirname, "..");
const SHOTS = path.join(OUT, ".tmp_shots");

const TAB_VALUES = [
  ["vacancies", "Вакансии"], ["data", "Результаты"], ["visualization", "Визуализация"],
  ["predictions", "Прогнозы"], ["articles", "Аналитика рынка"],
  ["scientific-trends", "Научные тренды"], ["teacher", "Преподавательский анализ"],
  ["students", "Студенты"], ["monitoring", "Мониторинг"], ["logs", "Логи"],
  ["admin", "Админ"], ["student", "Мои запросы"], ["taxonomy", "Таксономия"],
  ["profile", "Мой профиль"], ["help", "Помощь"],
];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function withTimeout(promise, ms, label) {
  let timer;
  return Promise.race([
    promise.finally(() => clearTimeout(timer)),
    new Promise((_, rej) => { timer = setTimeout(() => rej(new Error(`timeout ${ms}ms: ${label}`)), ms); }),
  ]);
}

async function httpJson(url) {
  const res = await fetch(url);
  return res.json();
}

class Client {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    this.events = [];
    ws.on("message", (data) => {
      const msg = JSON.parse(data.toString());
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      } else if (msg.method) {
        this.events.push(msg);
      }
    });
  }
  send(method, params = {}, ms = 20000) {
    const id = ++this.id;
    this.ws.send(JSON.stringify({ id, method, params }));
    return withTimeout(new Promise((resolve, reject) => this.pending.set(id, { resolve, reject })), ms, method);
  }
  async evaluate(expression, ms = 20000) {
    const r = await this.send("Runtime.evaluate", { expression, returnByValue: true, awaitPromise: true }, ms);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  }
}

const PROBE = `(() => {
  const de = document.documentElement;
  const grads = Array.from(document.querySelectorAll('*')).filter(el =>
    (getComputedStyle(el).backgroundImage || '').includes('gradient')).length;
  const small = Array.from(document.querySelectorAll('p,span,div,li,td'))
    .filter(el => el.children.length === 0 && (el.textContent || '').trim().length > 3)
    .filter(el => parseFloat(getComputedStyle(el).fontSize) < 12).length;
  const overflowers = Array.from(document.querySelectorAll('*')).filter(el => {
    const r = el.getBoundingClientRect();
    return r.width > 0 && (r.right > de.clientWidth + 2 || r.left < -2);
  }).length;
  const heading = document.querySelector('h1,h2') ? document.querySelector('h1,h2').innerText : null;
  return {
    hScroll: de.scrollWidth > de.clientWidth,
    scrollW: de.scrollWidth,
    clientW: de.clientWidth,
    gradients: grads,
    smallFont: small,
    overflowers,
    buttons: document.querySelectorAll('button').length,
    heading,
    text: (document.body.innerText || '').slice(0, 160).replace(/\\n+/g, ' | '),
  };
})()`;

(async () => {
  fs.mkdirSync(SHOTS, { recursive: true });
  const targets = await httpJson(`${CDP}/json/list`);
  const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
  if (!page) { console.error("no page target"); process.exit(1); }

  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false, maxPayload: 512 * 1024 * 1024 });
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });
  const c = new Client(ws);

  await c.send("Page.enable");
  await c.send("Runtime.enable");
  await c.send("Network.enable");
  console.log("cdp ready");

  await c.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  // Логин через API из Node, затем cookie ставим прямо в браузер через CDP.
  // Логин в контексте страницы ненадёжен: reload во время evaluate рвёт
  // соединение ("Inspected target navigated or closed").
  const loginEmail = process.env.SMOKE_EMAIL || "admin@compare-competencies.local";
  if (!process.env.SMOKE_PASSWORD) {
    console.error("Укажите SMOKE_PASSWORD (пароль админа) в окружении");
    process.exit(1);
  }
  const loginRes = await fetch("http://localhost:8000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: loginEmail, password: process.env.SMOKE_PASSWORD }),
  });
  if (!loginRes.ok) { console.error("login API failed:", loginRes.status); process.exit(1); }
  const session = await loginRes.json();
  if (!session.token) {
    console.error("login API не выдал токен:", JSON.stringify(session));
    process.exit(1);
  }
  console.log("login API ok, role =", session.role);

  await c.send("Network.setCookie", {
    name: "token",
    value: session.token,
    domain: "localhost",
    path: "/",
    httpOnly: true,
    sameSite: "Lax",
  });

  await c.send("Page.navigate", { url: APP });
  await sleep(7000);

  const whoami = await c.evaluate(`document.body.innerText.slice(0, 80).replace(/\\n+/g, " | ")`);
  console.log("after reload:", whoami);

  if (/Войдите в систему/.test(whoami)) {
    console.error("session not applied");
    process.exit(1);
  }

  const report = {};
  for (const [tab, label] of TAB_VALUES) {
    c.events.length = 0;
    const started = Date.now();
    let switched = "no";
    let probe = null;
    let shotOk = false;
    try {
      // Вкладки рисует NavGroup: одиночная группа - кнопка, группа из
      // нескольких - Radix DropdownMenu, пункты видны только после открытия.
      switched = await c.evaluate(`(async () => {
        const label = ${JSON.stringify(label)};
        const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
        const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
        // Radix открывает меню по pointerdown, обычный click не срабатывает.
        const realClick = (el) => {
          const opts = { bubbles: true, cancelable: true, button: 0, pointerId: 1, isPrimary: true };
          el.dispatchEvent(new PointerEvent('pointerdown', opts));
          el.dispatchEvent(new PointerEvent('pointerup', opts));
          el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
        };
        const allButtons = () => Array.from(document.querySelectorAll('button'));
        const direct = allButtons().find((b) => norm(b.innerText) === label && !b.hasAttribute('aria-haspopup'));
        if (direct) { realClick(direct); return 'direct'; }
        const triggers = allButtons().filter((b) => b.hasAttribute('aria-haspopup') || b.hasAttribute('data-state'));
        for (const t of triggers) {
          realClick(t);
          await sleep(400);
          const item = Array.from(document.querySelectorAll('[role="menuitem"],[role="menuitemradio"],[role="option"]'))
            .find((el) => norm(el.innerText) === label);
          if (item) { realClick(item); return 'menu'; }
          document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }));
          await sleep(150);
        }
        return 'not-found';
      })()`, 25000);
    } catch (e) { switched = `err: ${String(e.message).slice(0, 60)}`; }

    await sleep(3500);

    try { probe = await c.evaluate(PROBE, 15000); }
    catch (e) { probe = { error: String(e.message).slice(0, 120) }; }

    try {
      const img = await c.send("Page.captureScreenshot", { format: "png", captureBeyondViewport: false }, 25000);
      fs.writeFileSync(path.join(SHOTS, `${tab}.png`), Buffer.from(img.data, "base64"));
      shotOk = true;
    } catch (e) { /* скриншот не критичен */ }

    const errors = [...new Set(c.events
      .filter((e) => e.method === "Runtime.exceptionThrown" || (e.method === "Runtime.consoleAPICalled" && e.params.type === "error"))
      .map((e) => e.params.exceptionDetails
        ? (e.params.exceptionDetails.text + " " + (e.params.exceptionDetails.exception?.description || "")).slice(0, 150)
        : (e.params.args || []).map((a) => a.value || a.description).join(" ").slice(0, 150)))];

    const bad = [...new Set(c.events
      .filter((e) => e.method === "Network.responseReceived" && e.params.response.status >= 400)
      .map((e) => `${e.params.response.status} ${e.params.response.url.replace(APP, "").slice(0, 80)}`))];

    report[tab] = { switched, probe, errors, badRequests: bad, shot: shotOk, ms: Date.now() - started };
    console.log(`\n== ${tab} switched=${switched} shot=${shotOk} ${Date.now() - started}ms`);
    console.log("   ", JSON.stringify(probe));
    if (errors.length) console.log("    ERR:", errors.slice(0, 3));
    if (bad.length) console.log("    HTTP:", bad.slice(0, 4));
  }

  fs.writeFileSync(path.join(OUT, ".tmp_visual_report.json"), JSON.stringify(report, null, 2), "utf-8");
  console.log("\ndone: .tmp_visual_report.json + .tmp_shots/");
  ws.close();
  process.exit(0);
})().catch((e) => { console.error("FATAL", e.message); process.exit(1); });
