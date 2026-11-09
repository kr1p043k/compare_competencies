/**
 * Диагностика вкладки «Прогнозы»: что реально показывает список и какие
 * запросы рвутся на транспортном уровне.
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
  const consoleErrors = [];
  const failed = [];
  const apiCalls = [];
  ws.on("message", (d) => {
    const m = JSON.parse(d.toString());
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); return; }
    if (m.method === "Runtime.consoleAPICalled" && ["error", "warning"].includes(m.params.type)) {
      consoleErrors.push(m.params.args.map((a) => a.value ?? a.description).join(" ").slice(0, 200));
    }
    if (m.method === "Runtime.exceptionThrown") {
      consoleErrors.push("EXC " + (m.params.exceptionDetails.exception?.description || m.params.exceptionDetails.text).slice(0, 200));
    }
    if (m.method === "Network.loadingFailed") {
      failed.push(`${m.params.errorText} type=${m.params.type} canceled=${m.params.canceled}`);
    }
    if (m.method === "Network.responseReceived" && m.params.response.url.includes("/api/")) {
      apiCalls.push(`${m.params.response.status} ${m.params.response.url.replace(APP, "")}`);
    }
  });
  const send = (method, params = {}, ms = 30000) => new Promise((res, rej) => {
    const i = ++id;
    const t = setTimeout(() => rej(new Error(`timeout ${method}`)), ms);
    pending.set(i, (r) => { clearTimeout(t); res(r); });
    ws.send(JSON.stringify({ id: i, method, params }));
  });
  const evaluate = async (expr, ms = 30000) => {
    const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true }, ms);
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text);
    return r.result.value;
  };

  await send("Page.enable");
  await send("Runtime.enable");
  await send("Network.enable");

  const login = await fetch("http://localhost:8000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: "admin@compare-competencies.local", password: "admin" }),
  });
  const session = await login.json();

  await send("Page.navigate", { url: APP });
  await sleep(2500);
  await evaluate(`localStorage.setItem("auth", ${JSON.stringify(JSON.stringify(session))})`);

  apiCalls.length = 0; failed.length = 0; consoleErrors.length = 0;
  await send("Page.navigate", { url: `${APP}/#/predictions` });
  await sleep(9000);

  const snap = async (label) => ({
    label,
    listText: await evaluate(`(document.body.innerText.match(/(Топ растущих навыков[\\s\\S]{0,400})/) || [""])[0]`),
  });

  const out = [await snap("после загрузки")];

  // переключение на 3 месяца
  const switched = await evaluate(`(async () => {
    const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
    const norm = (s) => (s || '').replace(/\\s+/g, ' ').trim();
    const realClick = (el) => {
      const o = { bubbles: true, cancelable: true, button: 0, pointerId: 1, isPrimary: true };
      el.dispatchEvent(new PointerEvent('pointerdown', o));
      el.dispatchEvent(new PointerEvent('pointerup', o));
      el.dispatchEvent(new MouseEvent('click', { bubbles: true, cancelable: true }));
    };
    const trigger = Array.from(document.querySelectorAll('button')).find((b) => norm(b.innerText) === '1 месяц');
    if (!trigger) return 'no-trigger';
    realClick(trigger); await sleep(400);
    const opt = Array.from(document.querySelectorAll('[role="option"],[role="menuitem"]')).find((el) => norm(el.innerText) === '3 месяца');
    if (!opt) return 'no-option';
    realClick(opt); return 'ok';
  })()`);
  await sleep(6000);
  out.push(await snap("после 3 месяца"));

  console.log(JSON.stringify({
    switch: switched,
    snapshots: out,
    api: apiCalls,
    failedRequests: failed,
    console: consoleErrors,
  }, null, 2));
  ws.close();
  process.exit(0);
})().catch((e) => { console.error("FATAL", e.message); process.exit(1); });
