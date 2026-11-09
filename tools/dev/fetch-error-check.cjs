/**
 * Ловит точную причину «Нет связи с сервером»: оборачивает window.fetch и
 * пишет имя/сообщение каждой отклонённой попытки.
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
  await send("Page.addScriptToEvaluateOnNewDocument", {
    source: `
      window.__fetchErrs = [];
      const orig = window.fetch;
      window.fetch = function (...args) {
        const url = typeof args[0] === 'string' ? args[0] : (args[0] && args[0].url);
        return orig.apply(this, args).catch((e) => {
          window.__fetchErrs.push({ url, name: e && e.name, msg: e && e.message, signalAborted: !!(args[1] && args[1].signal && args[1].signal.aborted) });
          throw e;
        });
      };
    `,
  });

  const login = await fetch("http://localhost:8000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: "admin@compare-competencies.local", password: "admin" }),
  });
  const session = await login.json();

  await send("Page.navigate", { url: APP });
  await sleep(2500);
  await evaluate(`localStorage.setItem("auth", ${JSON.stringify(JSON.stringify(session))})`);

  await send("Page.navigate", { url: `${APP}/#/predictions` });
  await sleep(9000);

  console.log(JSON.stringify({
    fetchErrors: await evaluate(`window.__fetchErrs`),
    errorBanner: await evaluate(`(document.body.innerText.match(/Нет связи с сервером|Некорректн|Сервис временно/) || ["нет ошибки"])[0]`),
  }, null, 2));
  ws.close();
  process.exit(0);
})().catch((e) => { console.error("FATAL", e.message); process.exit(1); });
