const path = require("path");
const { createRequire } = require("module");

const frontendRequire = createRequire(path.resolve(__dirname, "..", "frontend", "package.json"));
const WebSocket = frontendRequire("ws");

const CDP = "http://127.0.0.1:9222";
const APP = "http://localhost:3000";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const targets = await (await fetch(`${CDP}/json/list`)).json();
  const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false, maxPayload: 64 * 1024 * 1024 });
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

  const login = await fetch("http://localhost:8000/api/auth/login", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ email: "admin@compare-competencies.local", password: "admin" }),
  });
  const session = await login.json();
  await send("Page.navigate", { url: APP });
  await sleep(2500);
  await evaluate(`localStorage.setItem("auth", ${JSON.stringify(JSON.stringify(session))})`);
  await send("Page.navigate", { url: `${APP}/#/teacher` });
  await sleep(9000);

  const out = await evaluate(`(() => {
    const bad = [];
    for (const el of document.querySelectorAll('p,span,div,li,td')) {
      if (el.children.length) continue;
      const txt = (el.textContent || '').trim();
      if (txt.length <= 3) continue;
      const fs = parseFloat(getComputedStyle(el).fontSize);
      if (fs < 12) {
        bad.push({
          fs,
          cls: (el.className || '').toString().slice(0, 110),
          tag: el.tagName.toLowerCase(),
          text: txt.slice(0, 60),
          inline: (el.getAttribute('style') || '').slice(0, 80),
        });
      }
    }
    return { count: bad.length, items: bad.slice(0, 30) };
  })()`);

  console.log(JSON.stringify(out, null, 2));
  ws.close();
  process.exit(0);
})().catch((e) => { console.error("FATAL", e.message); process.exit(1); });
