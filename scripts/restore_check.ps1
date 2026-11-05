<#
.SYNOPSIS
    Проверка восстановимости бэкапа PostgreSQL в временную БД cc_restore_check.

.DESCRIPTION
    Берёт последний *.sql из data/backup (или data/backups), создаёт временную
    базу cc_restore_check, восстанавливает в неё дамп, проверяет что таблиц > 0,
    и удаляет базу. Рабочая база compare_competencies НЕ затрагивается.

    По умолчанию - dry-run: скрипт только печатает план и ничего не делает.
    Реальное восстановление запускается только с флагом -Confirm.

    Параметры подключения берутся из переменных окружения POSTGRES_*, при их
    отсутствии - из файла .env в корне проекта.

.PARAMETER Dir
    Каталог с бэкапами. По умолчанию data/backup / data/backups.

.PARAMETER Confirm
    Выполнить восстановление по-настоящему (иначе dry-run).

.PARAMETER Keep
    Не удалять cc_restore_check после проверки (для разбора вручную).

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File scripts\restore_check.ps1
    powershell -ExecutionPolicy Bypass -File scripts\restore_check.ps1 -Confirm
#>
[CmdletBinding()]
param(
    [string]$Dir = "",
    [switch]$Confirm,
    [switch]$Keep
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$TempDb = "cc_restore_check"

$script:Problems = New-Object System.Collections.Generic.List[string]
function Write-Step { param([string]$Text) Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "  [OK]   $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "  [WARN] $Text" -ForegroundColor Yellow }
function Write-Bad  { param([string]$Text) Write-Host "  [FAIL] $Text" -ForegroundColor Red; $script:Problems.Add($Text) }
function Write-Dim  { param([string]$Text) Write-Host "        $Text" -ForegroundColor DarkYellow }

# --- .env (простые KEY=VALUE, без кавычек) ---------------------------------
function Import-DotEnv {
    $path = Join-Path $root ".env"
    if (-not (Test-Path -LiteralPath $path)) { return }
    foreach ($line in Get-Content -LiteralPath $path -Encoding utf8) {
        if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
        $idx = $line.IndexOf('=')
        $key = $line.Substring(0, $idx).Trim()
        $val = $line.Substring($idx + 1).Trim()
        if ($key -and -not (Test-Path ("Env:" + $key))) {
            Set-Item -Path ("Env:" + $key) -Value $val
        }
    }
}

function Resolve-PgTool {
    param([string]$Name)
    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $bases = Get-ChildItem -Path "C:\Program Files\PostgreSQL" -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($b in $bases) {
        $candidate = Join-Path $b.FullName "bin\$Name.exe"
        if (Test-Path -LiteralPath $candidate) { return $candidate }
    }
    return $null
}

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
    foreach ($candidate in @("data\backup", "data\backups")) {
        $full = Join-Path $root $candidate
        if (Test-Path -LiteralPath $full) { return $full }
    }
    return $null
}

function Invoke-Psql {
    param(
        [string]$Psql,
        [string]$DbName,
        [string[]]$ExtraArgs,
        [string]$Sql = ""
    )
    $psqlArgs = @("-w", "-h", $script:PgHost, "-p", $script:PgPort, "-U", $script:PgUser)
    if ($DbName) { $psqlArgs += @("-d", $DbName) }
    if ($ExtraArgs) { $psqlArgs += $ExtraArgs }
    if ($Sql) { $psqlArgs += @("-tA", "-c", $Sql) }
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $out = & $Psql @psqlArgs 2>&1
        return [pscustomobject]@{ Code = $LASTEXITCODE; Out = ($out | Out-String).Trim() }
    } finally {
        $ErrorActionPreference = $prev
    }
}

# --- main -------------------------------------------------------------------
Import-DotEnv

$script:PgHost = if ($env:POSTGRES_HOST) { $env:POSTGRES_HOST } else { "localhost" }
$script:PgPort = if ($env:POSTGRES_PORT) { $env:POSTGRES_PORT } else { "5432" }
$script:PgUser = if ($env:POSTGRES_USER) { $env:POSTGRES_USER } else { "postgres" }
$pgPassword     = $env:POSTGRES_PASSWORD

Write-Host "########## RESTORE CHECK ##########" -ForegroundColor Magenta

$psql = Resolve-PgTool -Name "psql"
if (-not $psql) { Write-Bad "psql не найден (установите PostgreSQL client или добавьте в PATH)"; exit 1 }
$pgRestore = Resolve-PgTool -Name "pg_restore"
Write-Dim "psql: $psql"

$backupDir = Resolve-BackupDir -Explicit $Dir
if (-not $backupDir) { Write-Bad "каталог бэкапов не найден (ожидался data/backup)"; exit 1 }
$dump = @(Get-ChildItem -LiteralPath $backupDir -Filter *.sql -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending)[0]
if ($null -eq $dump) {
    Write-Bad "в каталоге $backupDir нет ни одного *.sql"
    Write-Dim "создайте дамп: python -m src.cli backup"
    exit 1
}

Write-Step "План"
Write-Dim ("дамп:        {0} ({1} MB)" -f $dump.FullName, [math]::Round($dump.Length / 1MB, 2))
Write-Dim ("сервер:      {0}:{1} / user={2}" -f $script:PgHost, $script:PgPort, $script:PgUser)
Write-Dim ("временная БД: {0}" -f $TempDb)
Write-Dim ("после проверки: {0}" -f $(if ($Keep) { "БД останется (-Keep)" } else { "БД будет удалена" }))

if (-not $Confirm) {
    Write-Host "`nDRY-RUN: восстановление не выполнялось." -ForegroundColor Yellow
    Write-Host "Для реального запуска добавьте -Confirm." -ForegroundColor Yellow
    exit 0
}

$isCustomFormat = $false
try {
    $head = Get-Content -LiteralPath $dump.FullName -TotalCount 1 -Encoding ascii -ErrorAction Stop
    if ($head -and $head.StartsWith("PGDMP")) { $isCustomFormat = $true }
} catch { }

$prevPgPassword = $env:PGPASSWORD
if ($pgPassword) { $env:PGPASSWORD = $pgPassword }

$dbCreated = $false
try {
    Write-Step "Создание временной БД"
    [void](Invoke-Psql -Psql $psql -DbName "postgres" -ExtraArgs @("-v", "ON_ERROR_STOP=1") `
        -Sql "DROP DATABASE IF EXISTS $TempDb;")
    $create = Invoke-Psql -Psql $psql -DbName "postgres" -ExtraArgs @("-v", "ON_ERROR_STOP=1") `
        -Sql "CREATE DATABASE $TempDb;"
    if ($create.Code -ne 0) {
        Write-Bad ("не удалось создать {0} (exit {1})" -f $TempDb, $create.Code)
        Write-Dim $create.Out
        exit 1
    }
    $dbCreated = $true
    Write-Ok "БД $TempDb создана"

    Write-Step "Восстановление дампа"
    if ($isCustomFormat) {
        if (-not $pgRestore) { Write-Bad "pg_restore не найден для custom-дампа"; exit 1 }
        $prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
        try {
            $out = & $pgRestore "--no-owner" "--dbname=$TempDb" "-h" $script:PgHost "-p" $script:PgPort `
                "-U" $script:PgUser $dump.FullName 2>&1
            $rc = $LASTEXITCODE
        } finally { $ErrorActionPreference = $prev }
        if ($rc -ne 0) { Write-Warn ("pg_restore завершился с кодом {0} (часто из-за warnings)" -f $rc) }
        else { Write-Ok "pg_restore завершён" }
    } else {
        $rc = Invoke-Psql -Psql $psql -DbName $TempDb -ExtraArgs @("-v", "ON_ERROR_STOP=1", "-f", $dump.FullName)
        if ($rc.Code -ne 0) {
            Write-Bad ("psql -f завершился с кодом {0}" -f $rc.Code)
            Write-Dim $rc.Out
            exit 1
        }
        Write-Ok "дамп применён"
    }

    Write-Step "Проверка таблиц"
    $count = Invoke-Psql -Psql $psql -DbName $TempDb `
        -Sql "SELECT count(*) FROM information_schema.tables WHERE table_schema NOT IN ('pg_catalog','information_schema');"
    $n = 0
    if ($count.Code -eq 0 -and $count.Out -match '^\d+$') { $n = [int]$count.Out }
    if ($n -gt 0) { Write-Ok ("таблиц в восстановленной БД: {0}" -f $n) }
    else { Write-Bad ("таблиц не найдено (raw: '{0}', exit {1})" -f $count.Out, $count.Code) }
} finally {
    if ($dbCreated -and -not $Keep) {
        Write-Step "Удаление временной БД"
        $drop = Invoke-Psql -Psql $psql -DbName "postgres" -ExtraArgs @("-v", "ON_ERROR_STOP=1") `
            -Sql "DROP DATABASE IF EXISTS $TempDb;"
        if ($drop.Code -eq 0) { Write-Ok "БД $TempDb удалена" }
        else { Write-Warn ("не удалось удалить {0}" -f $TempDb); Write-Dim $drop.Out }
    } elseif ($dbCreated -and $Keep) {
        Write-Warn ("БД {0} оставлена (-Keep)" -f $TempDb)
    }
    if ($null -ne $prevPgPassword) { $env:PGPASSWORD = $prevPgPassword } else { Remove-Item Env:PGPASSWORD -ErrorAction SilentlyContinue }
}

Write-Host "`n########## RESULT ##########" -ForegroundColor Magenta
if ($script:Problems.Count -eq 0) {
    Write-Host "GATE: GREEN - дамп восстановлен, таблицы на месте" -ForegroundColor Green
    exit 0
}
Write-Host ("GATE: RED - проблем: {0}" -f $script:Problems.Count) -ForegroundColor Red
$script:Problems | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
exit 1
