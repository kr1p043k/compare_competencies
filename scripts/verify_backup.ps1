<#
.SYNOPSIS
    Проверка последнего дампа PostgreSQL из data/backup/.

.DESCRIPTION
    Берёт самый свежий *.sql в каталоге бэкапов и проверяет:
      1. файл существует и размер > 0;
      2. в дампе есть маркеры plain-SQL дампа PostgreSQL
         ("PostgreSQL database dump") и данных (COPY / INSERT INTO);
      3. считает число таблиц (по CREATE TABLE, с запасным счётом по COPY).
    Возвращает exit code 0 (OK) или 1 (проблема) и печатает сводку.

    Скрипт только читает: ничего не восстанавливает и не меняет.
    Для восстановления в тестовую БД - scripts\restore_check.ps1.

.PARAMETER Dir
    Каталог с бэкапами. По умолчанию data/backup, при его отсутствии или
    пустоте берётся data/backups (совместимость с src/cli/backup_db.py).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\verify_backup.ps1
    powershell -ExecutionPolicy Bypass -File scripts\verify_backup.ps1 -Dir data\backups
#>
[CmdletBinding()]
param(
    [string]$Dir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$script:Problems = New-Object System.Collections.Generic.List[string]

function Write-Step { param([string]$Text) Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "  [OK]   $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "  [WARN] $Text" -ForegroundColor Yellow }
function Write-Bad  { param([string]$Text) Write-Host "  [FAIL] $Text" -ForegroundColor Red; $script:Problems.Add($Text) }
function Write-Dim  { param([string]$Text) Write-Host "        $Text" -ForegroundColor DarkYellow }

# 1. где искать бэкапы ------------------------------------------------------
function Resolve-BackupDir {
    param([string]$Explicit)
    if ($Explicit) {
        if (Test-Path -LiteralPath $Explicit) { return (Resolve-Path -LiteralPath $Explicit).Path }
        return $null
    }
    foreach ($candidate in @("data\backup", "data\backups")) {
        $full = Join-Path $root $candidate
        if (Test-Path -LiteralPath $full) {
            $hasSql = @(Get-ChildItem -LiteralPath $full -Filter *.sql -File -ErrorAction SilentlyContinue)
            if ($hasSql.Count -gt 0) { return $full }
        }
    }
    # Каталог есть, но пустой: вернём первый существующий, чтобы сказать "пусто".
    foreach ($candidate in @("data\backup", "data\backups")) {
        $full = Join-Path $root $candidate
        if (Test-Path -LiteralPath $full) { return $full }
    }
    return $null
}

function Get-LatestDump {
    param([string]$BackupDir)
    $sql = @(Get-ChildItem -LiteralPath $BackupDir -Filter *.sql -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    if ($sql.Count -eq 0) { return $null }
    return $sql[0]
}

# 2. main -------------------------------------------------------------------
Write-Host "########## BACKUP VERIFY ##########" -ForegroundColor Magenta

Write-Step "Поиск последнего дампа"
$backupDir = Resolve-BackupDir -Explicit $Dir
if (-not $backupDir) {
    Write-Bad "каталог бэкапов не найден (ожидался data/backup)"
    Write-Host "`nRESULT: FAIL - проверять нечего" -ForegroundColor Red
    exit 1
}
Write-Dim "каталог: $backupDir"

$dump = Get-LatestDump -BackupDir $backupDir
if ($null -eq $dump) {
    Write-Bad "в каталоге $backupDir нет ни одного *.sql"
    Write-Dim "создайте дамп: python -m src.cli backup"
    Write-Host "`nRESULT: FAIL - бэкапов нет" -ForegroundColor Red
    exit 1
}
Write-Ok ("последний дамп: {0}" -f $dump.Name)
Write-Dim ("изменён: {0}" -f $dump.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))

# 3. размер -----------------------------------------------------------------
Write-Step "Размер"
$sizeBytes = $dump.Length
$sizeMb = [math]::Round($sizeBytes / 1MB, 2)
if ($sizeBytes -le 0) {
    Write-Bad "файл пустой (0 байт)"
} else {
    Write-Ok ("размер: {0} байт ({1} MB)" -f $sizeBytes, $sizeMb)
}

# 4. маркеры ----------------------------------------------------------------
Write-Step "Маркеры дампа"
$isCustomFormat = $false
try {
    $head = Get-Content -LiteralPath $dump.FullName -TotalCount 1 -Encoding ascii -ErrorAction Stop
    if ($head -and $head.StartsWith("PGDMP")) { $isCustomFormat = $true }
} catch { }

if ($isCustomFormat) {
    Write-Warn "дамп в custom-формате (pg_dump -Fc): текстовые маркеры неприменимы"
    Write-Dim "для такого дампа используйте pg_restore / scripts\restore_check.ps1"
} else {
    $markerDump = Select-String -LiteralPath $dump.FullName -Pattern "PostgreSQL database dump" -SimpleMatch -Quiet
    if ($markerDump) { Write-Ok "маркер 'PostgreSQL database dump' найден" }
    else { Write-Bad "нет маркера 'PostgreSQL database dump' (не plain-SQL дамп?)" }

    $markerCopy = Select-String -LiteralPath $dump.FullName -Pattern "^COPY " -Quiet
    $markerInsert = Select-String -LiteralPath $dump.FullName -Pattern "INSERT INTO" -SimpleMatch -Quiet
    if ($markerCopy) {
        Write-Ok "найдены COPY-блоки (данные)"
    } elseif ($markerInsert) {
        Write-Ok "найдены INSERT INTO (данные)"
    } else {
        Write-Bad "нет ни COPY, ни INSERT INTO - дамп без данных?"
    }
}

# 5. таблицы ----------------------------------------------------------------
Write-Step "Таблицы"
$tableCount = 0
if (-not $isCustomFormat) {
    $createMatches = @(Select-String -LiteralPath $dump.FullName -Pattern "^CREATE TABLE " -CaseSensitive:$false)
    $tableCount = $createMatches.Count
    if ($tableCount -eq 0) {
        $copyMatches = @(Select-String -LiteralPath $dump.FullName -Pattern "^COPY " -CaseSensitive:$false)
        $tableCount = $copyMatches.Count
        if ($tableCount -gt 0) { Write-Warn "CREATE TABLE не найден, таблицы посчитаны по COPY" }
    }
}
if ($isCustomFormat) {
    Write-Warn "custom-формат: число таблиц не считается без pg_restore"
} elseif ($tableCount -gt 0) {
    Write-Ok ("таблиц в дампе: {0}" -f $tableCount)
} else {
    Write-Bad "таблиц не найдено (0 CREATE TABLE / COPY)"
}

# 6. итог -------------------------------------------------------------------
Write-Host "`n########## RESULT ##########" -ForegroundColor Magenta
if ($script:Problems.Count -eq 0) {
    Write-Host ("GATE: GREEN - дамп {0} выглядит пригодным" -f $dump.Name) -ForegroundColor Green
    exit 0
}
Write-Host ("GATE: RED - проблем: {0}" -f $script:Problems.Count) -ForegroundColor Red
$script:Problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
exit 1
