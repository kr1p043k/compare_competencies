/** Найти элементы с шрифтом < 12px на конкретной вкладке. */
const fs = require("fs");
const path = require("path");
const { createRequire } = require("module");

const frontendRequire = createRequire(path.resolve(__dirname, "..", "frontend", "package.json"));
const WebSocket = frontendRequire("ws");

const CDP = "http://127.0.0.1:9222";
const APP = "http://localhost:3000";
const LABEL = process.argv[2] || "Преподавательский анализ";

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

(async () => {
  const targets = await (await fetch(`${CDP}/json/list`)).json();
  const page = targets.find((t) => t.type === "page" && t.webSocketDebuggerUrl);
  const ws = new WebSocket(page.webSocketDebuggerUrl, { perMessageDeflate: false });
  await new Promise((res, rej) => { ws.on("open", res); ws.on("error", rej); });

  let id = 0;
  const pending = new Map();
  ws.on("message", (d) => {
    const m = JSON.parse(d.toString());
    if (m.id && pending.has(m.id)) { pending.get(m.id)(m.result); pending.delete(m.id); }
  });
  const send = (method, params = {}) => new Promise((res) => { const i = ++id; pending.set(i, res); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expr) => (await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true })).result.value;

  await send("Runtime.enable");
  await send("Page.enable");

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
      const item = Array.from(document.querySelectorAll('[role="menuitem"]')).find((el) => norm(el.innerText) === ${JSON.stringify(LABEL)});
      if (item) { realClick(item); return 'ok'; }
      document.body.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true })); await sleep(120);
    }
    return 'not-found';
  })()`);
  await sleep(3500);

  const offenders = await evaluate(`(() => {
    const out = [];
    document.querySelectorAll('*').forEach((el) => {
      if (el.children.length) return;
      const t = (el.textContent || '').trim();
      if (t.length < 3) return;
      const cs = getComputedStyle(el);
      const size = parseFloat(cs.fontSize);
      if (size < 12) out.push({ size, tag: el.tagName, cls: (el.className || '').toString().slice(0, 90), text: t.slice(0, 50) });
    });
    return out.slice(0, 20);
  })()`);

  console.log(JSON.stringify(offenders, null, 2));
  ws.close();
  process.exit(0);
})().catch((e) => { console.error(e.message); process.exit(1); });
