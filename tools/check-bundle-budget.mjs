#!/usr/bin/env node
/**
 * Бюджет бандла фронтенда (BACKLOG 6.5.4).
 *
 * Считает gzip-размер JS/CSS из frontend/dist/assets и сравнивает с бюджетом.
 * Без гейта размер растёт незаметно: подключают тяжёлую либу (recharts, motion),
 * и первоначальная загрузка тихо жирнеет.
 *
 * Запуск: node tools/check-bundle-budget.mjs [--dist <путь>] [--json]
 * Код возврата: 0 - в бюджете, 1 - превышен.
 */
import { gzipSync } from "node:zlib";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Корень репозитория считаем от самого скрипта: он лежит в <repo>/tools,
// поэтому не зависим от cwd (npm run запускает из frontend).
const REPO_ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const BUDGET_KB = {
  // Первая загрузка: суммарный JS+CSS (gzip).
  total: 520,
  // Самый крупный JS-чанк (gzip) - чтобы один вендор не тянул всё за собой.
  largestJs: 300,
};

function arg(name, fallback) {
  const i = process.argv.indexOf(name);
  return i !== -1 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const dist = path.resolve(REPO_ROOT, arg("--dist", "frontend/dist"));
const assets = path.join(dist, "assets");
const asJson = process.argv.includes("--json");

if (!fs.existsSync(assets)) {
  console.error(`Нет ${assets}. Сначала соберите: npm --prefix frontend run build`);
  process.exit(1);
}

const rows = [];
for (const name of fs.readdirSync(assets)) {
  if (!/\.(js|css)$/.test(name)) continue;
  const buf = fs.readFileSync(path.join(assets, name));
  rows.push({ name, raw: buf.length, gz: gzipSync(buf).length });
}

const gz = (b) => Math.round(b / 1024);
const totalGz = rows.reduce((s, r) => s + r.gz, 0);
const jsRows = rows.filter((r) => r.name.endsWith(".js"));
const largestJs = jsRows.reduce((m, r) => Math.max(m, r.gz), 0);

const problems = [];
if (gz(totalGz) > BUDGET_KB.total) problems.push(`total ${gz(totalGz)}KB > ${BUDGET_KB.total}KB`);
if (gz(largestJs) > BUDGET_KB.largestJs) problems.push(`largest JS ${gz(largestJs)}KB > ${BUDGET_KB.largestJs}KB`);

if (asJson) {
  console.log(JSON.stringify({ total: gz(totalGz), largestJs: gz(largestJs), budget: BUDGET_KB, problems, files: rows.map((r) => ({ name: r.name, gz: gz(r.gz) })) }, null, 2));
} else {
  for (const r of rows.sort((a, b) => b.gz - a.gz).slice(0, 10)) {
    console.log(`  ${String(gz(r.gz)).padStart(5)} KB  ${r.name}`);
  }
  console.log(`\nИтого (gzip): ${gz(totalGz)} KB / бюджет ${BUDGET_KB.total} KB`);
  console.log(`Крупнейший JS: ${gz(largestJs)} KB / бюджет ${BUDGET_KB.largestJs} KB`);
  console.log(problems.length ? `\nBUNDLE: ПРЕВЫШЕН - ${problems.join("; ")}` : "\nBUNDLE: в бюджете");
}

process.exit(problems.length ? 1 : 0);
