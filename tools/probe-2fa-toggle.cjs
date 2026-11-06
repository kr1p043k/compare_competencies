/**
 * Проверка тумблера 2FA в живом UI через CDP.
 *
 * Что делает: открывает #/admin, заходит в под-вкладку «Безопасность»,
 * нажимает переключатель, сверяет состояние с API, затем возвращает
 * переключатель в исходное (выключенное) положение.
 *
 * Требует: uvicorn :8000, vite :3000, chrome --remote-debugging-port=9222,
 * SMOKE_PASSWORD в окружении.
 */
const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");

const frontendRequire = createRequire(path.resolve(__dirname, "..", "frontend", "package.json"));
const WebSocket = frontendRequire("ws");

const CDP = "http://127.0.0.1:9222";
const APP = "http://localhost:3000";
const API = "http://127.0.0.1:8000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function httpJson(url) {
  const res = await fetch(url);
  return res.json();
}

class Client {
  constructor(ws) {
    this.ws = ws;
    this.id = 0;
    this.pending = new Map();
    ws.on("message", (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id);
        this.pending.delete(msg.id);
        msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result);
      }
    });
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }
  async evaluate(expression) {
    const r = await this.send("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text || "evaluate failed");
    return r.result.value;
  }
}

async function apiToggle(token, enabled) {
  const res = await fetch(`${API}/api/admin/security/2fa/settings`, {
    method: "PUT",
    headers: { "Content-Type": "application/json", Cookie: `token=${token}` },
    body: JSON.stringify({ enabled }),
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

async function apiState(token) {
  const res = await fetch(`${API}/api/admin/security/2fa/settings`, {
    headers: { Cookie: `token=${token}` },
  });
  return { status: res.status, body: await res.json().catch(() => ({})) };
}

(async () => {
  const email = process.env.SMOKE_EMAIL || "admin@compare-competencies.local";
  const password = process.env.SMOKE_PASSWORD;
  if (!password) {
    console.error("Укажите SMOKE_PASSWORD");
    process.exit(1);
  }

  const loginRes = await fetch(`${API}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email, password }),
  });
  if (!loginRes.ok) {
    console.error("login failed:", loginRes.status);
    process.exit(1);
  }
  const token = (await loginRes.json()).token;

  // Исходное состояние должно быть «выключено»: проверка это не меняет.
  const before = await apiState(token);
  console.log("API до:", JSON.stringify(before.body));

  const targets = await httpJson(`${CDP}/json/list`);
  const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
  if (!page) {
    console.error("no page target");
    process.exit(1);
  }
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false, maxPayload: 512 * 1024 * 1024 });
  await new Promise((res, rej) => {
    ws.on("open", res);
    ws.on("error", rej);
  });
  const c = new Client(ws);
  await c.send("Page.enable");
  await c.send("Runtime.enable");
  await c.send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  await c.send("Network.setCookie", {
    name: "token",
    value: token,
    domain: "localhost",
    path: "/",
    httpOnly: true,
    sameSite: "Lax",
  });

  await c.send("Page.navigate", { url: `${APP}/#/admin` });
  await sleep(7000);

  // Открыть под-вкладку «Безопасность» внутри админки.
  const opened = await c.evaluate(`(async () => {
    const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
    const realClick = (el) => {
      const opts = { bubbles: true, cancelable: true, button: 0, pointerId: 1, isPrimary: true };
      el.dispatchEvent(new PointerEvent("pointerdown", opts));
      el.dispatchEvent(new MouseEvent("mousedown", opts));
      el.dispatchEvent(new PointerEvent("pointerup", opts));
      el.dispatchEvent(new MouseEvent("mouseup", opts));
      el.dispatchEvent(new MouseEvent("click", opts));
    };
    const tabs = Array.from(document.querySelectorAll('[role="tab"],button'));
    const target = tabs.find((b) => norm(b.innerText) === "Безопасность");
    if (!target) return { ok: false, have: tabs.map((b) => norm(b.innerText)).filter(Boolean).slice(0, 12) };
    realClick(target);
    await new Promise((r) => setTimeout(r, 2500));
    return { ok: true, selected: target.getAttribute("data-state") || target.getAttribute("aria-selected") };
  })()`);
  console.log("вкладка Безопасность:", JSON.stringify(opened));

  const cardSeen = await c.evaluate(`(async () => {
    const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
    const text = norm(document.body.innerText);
    return { hasHeading: text.includes("Двухфакторная аутентификация"), snippet: text.slice(0, 400) };
  })()`);
  console.log("карточка:", JSON.stringify(cardSeen));

  // Нажать переключатель (включить).
  const clicked = await c.evaluate(`(async () => {
    const norm = (s) => (s || "").replace(/\\s+/g, " ").trim();
    const buttons = Array.from(document.querySelectorAll("button"));
    const candidates = buttons.filter((b) =>
      ["Включить", "Отключить"].some((t) => norm(b.innerText).includes(t)),
    );
    let container = null;
    let button = null;
    for (const candidate of candidates) {
      let parent = candidate.parentElement;
      for (let depth = 0; depth < 8 && parent; depth += 1) {
        if (norm(parent.innerText).includes("Двухфакторная аутентификация")) {
          container = parent;
          button = candidate;
          break;
        }
        parent = parent.parentElement;
      }
      if (button) break;
    }
    if (!container || !button) {
      return { ok: false, reason: "карточка не найдена", candidates: candidates.map((b) => norm(b.innerText)) };
    }
    const before = norm(button.innerText);
    button.click();
    await new Promise((r) => setTimeout(r, 2500));
    return { ok: true, clickedText: before, cardText: norm(container.innerText).slice(0, 160) };
  })()`);
  console.log("клик:", JSON.stringify(clicked));

  await sleep(1500);
  const afterOn = await apiState(token);
  console.log("API после клика:", JSON.stringify(afterOn.body));

  // Вернуть в исходное выключенное состояние.
  if (afterOn.body?.enabled) {
    const off = await apiToggle(token, false);
    console.log("вернул выключенное:", JSON.stringify(off));
  }
  const finalState = await apiState(token);
  console.log("API итог:", JSON.stringify(finalState.body));

  const shots = path.resolve(__dirname, "..", ".tmp_shots");
  fs.mkdirSync(shots, { recursive: true });
  const shot = await c.send("Page.captureScreenshot", { format: "png" });
  fs.writeFileSync(path.join(shots, "2fa-toggle.png"), Buffer.from(shot.data, "base64"));
  console.log("скриншот: .tmp_shots/2fa-toggle.png");

  ws.close();
  process.exit(0);
})();
