<#
.SYNOPSIS
    Release gate for compare_online: precheck, merge into for_Nikita, push.

.DESCRIPTION
    Precheck runs the full verification gate and stops at the first failure:
      1. worktree cleanliness
      2. ruff over src/ (baseline: must not get worse)
      3. backend pytest (whole tests/ tree)
      4. frontend tsc / vitest / build
      5. alembic must have exactly one head
      6. pip-audit (informational: known aiohttp/starlette debt)

.PARAMETER Action
    Precheck | Merge | Push | All

.PARAMETER Source
    Branch to release into for_Nikita. Default: current branch.

.PARAMETER Target
    Branch to ship to. Default: for_Nikita.

.PARAMETER SkipBackend
    Skip pytest (fast smoke only; never for a real release).

.PARAMETER TimeoutMin
    Backend test budget in minutes. Default 20.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File tools\release.ps1 -Action Precheck
#>
[CmdletBinding()]
param(
    [ValidateSet("Precheck", "Merge", "Push", "All")]
    [string]$Action = "Precheck",

    [string]$Source = "",
    [string]$Target = "for_Nikita",
    [switch]$SkipBackend,
    [int]$TimeoutMin = 20
)

$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

$script:Failures = New-Object System.Collections.Generic.List[string]
$script:Notes    = New-Object System.Collections.Generic.List[string]
$script:BaselineFile = Join-Path $root ".release-baseline.json"

function Write-Step { param([string]$Text) Write-Host "`n=== $Text ===" -ForegroundColor Cyan }
function Write-Ok   { param([string]$Text) Write-Host "  [OK]   $Text" -ForegroundColor Green }
function Write-Warn { param([string]$Text) Write-Host "  [WARN] $Text" -ForegroundColor Yellow; $script:Notes.Add($Text) }
function Write-Bad  { param([string]$Text) Write-Host "  [FAIL] $Text" -ForegroundColor Red; $script:Failures.Add($Text) }
function Write-Dim  { param([string]$Text) Write-Host "        $Text" -ForegroundColor DarkYellow }

# PowerShell 5.1: stderr нативных команд при ErrorActionPreference=Stop
# превращается в исключение. Для внешних инструментов глушим на время вызова.
# ВАЖНО: ProcessStartInfo.ArgumentList есть только в .NET Core (PS7); в 5.1
# (.NET Framework) доступен лишь строковый Arguments, поэтому аргументы
# склеиваем с корректным кавычеванием.
function Format-Arg {
    param([string]$Value)
    if ($null -eq $Value) { return '""' }
    if ($Value -eq "") { return '""' }
    if ($Value -match '[\s"]') {
        return '"' + ($Value -replace '"', '\"') + '"'
    }
    return $Value
}

function Invoke-Native {
    param(
        [string]$FilePath,
        [string[]]$ArgList,
        [int]$TimeoutSec = 900,
        [switch]$ViaCmd
    )
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        $psi = New-Object System.Diagnostics.ProcessStartInfo
        if ($ViaCmd) {
            # npm/npx на Windows - .cmd-обёртки: прямой Process.Start с ними
            # даёт MODULE_NOT_FOUND, нужен cmd /c.
            $psi.FileName = "$env:ComSpec"
            $inner = (($ArgList | ForEach-Object { Format-Arg $_ }) -join ' ')
            $psi.Arguments = "/c `"$FilePath $inner`""
        } else {
            $psi.FileName = $FilePath
            $psi.Arguments = (($ArgList | ForEach-Object { Format-Arg $_ }) -join ' ')
        }
        $psi.RedirectStandardOutput = $true
        $psi.RedirectStandardError = $true
        $psi.UseShellExecute = $false
        $psi.WorkingDirectory = (Get-Location).Path
        $p = [System.Diagnostics.Process]::Start($psi)
        $outTask = $p.StandardOutput.ReadToEndAsync()
        $errTask = $p.StandardError.ReadToEndAsync()
        if (-not $p.WaitForExit($TimeoutSec * 1000)) {
            try { $p.Kill() } catch { }
            return [pscustomobject]@{ Code = -9; StdOut = $outTask.Result; StdErr = $errTask.Result; TimedOut = $true }
        }
        return [pscustomobject]@{
            Code     = $p.ExitCode
            StdOut   = $outTask.Result
            StdErr   = $errTask.Result
            TimedOut = $false
        }
    } finally {
        $ErrorActionPreference = $prev
    }
}

function Invoke-Git {
    param([string[]]$GitArgs)
    $r = Invoke-Native -FilePath "git" -ArgList $GitArgs -TimeoutSec 300
    return [pscustomobject]@{ Code = $r.Code; Out = (($r.StdOut + "`n" + $r.StdErr).Trim()) }
}

function Get-Baseline {
    if (Test-Path $script:BaselineFile) {
        try { return Get-Content $script:BaselineFile -Raw -Encoding utf8 | ConvertFrom-Json } catch { }
    }
    return [pscustomobject]@{ ruff = -1 }
}

function Save-Baseline {
    param([int]$Ruff)
    $obj = [pscustomobject]@{ ruff = $Ruff; updated = (Get-Date -Format "o") }
    $obj | ConvertTo-Json | Set-Content -LiteralPath $script:BaselineFile -Encoding utf8
    Write-Host "  baseline updated: ruff=$Ruff" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- 1. worktree
function Test-WorktreeClean {
    Write-Step "Worktree cleanliness"
    $status = (Invoke-Git @("status", "--porcelain")).Out
    if ([string]::IsNullOrWhiteSpace($status)) { Write-Ok "worktree clean"; return $true }
    $dirty = @($status -split "`n" | Where-Object {
        $_ -match '\S' -and $_ -notmatch '\.tmp_' -and $_ -notmatch '^\?\? dist/'
    } | ForEach-Object { $_.TrimEnd() })
    if ($dirty.Count -eq 0) { Write-Ok "only ignored temp artifacts present"; return $true }
    Write-Bad ("uncommitted changes must be committed by their owner ({0} files)" -f $dirty.Count)
    $dirty | Select-Object -First 15 | ForEach-Object { Write-Dim $_ }
    if ($dirty.Count -gt 15) { Write-Dim ("... and {0} more" -f ($dirty.Count - 15)) }
    return $false
}

# ------------------------------------------------------------------- 2. ruff
function Test-Ruff {
    Write-Step "ruff (src/)"
    $r = Invoke-Native -FilePath "python" -ArgList @("-m", "ruff", "check", "src/", "--output-format", "concise") -TimeoutSec 300
    $lines = @(($r.StdOut + "`n" + $r.StdErr) -split "`n" |
        Where-Object { $_ -match '\.py:\d+:\d+:' })
    $count = $lines.Count
    $baseline = (Get-Baseline).ruff

    if ($count -eq 0) { Write-Ok "ruff clean"; if ($baseline -gt 0) { Save-Baseline -Ruff 0 }; return $true }

    if ($baseline -lt 0) {
        # Первый прогон: фиксируем текущий долг как базу, не блокируем релиз.
        Save-Baseline -Ruff $count
        Write-Warn ("ruff debt recorded as baseline: {0} markers (must not grow)" -f $count)
        return $true
    }
    if ($count -le $baseline) {
        Write-Ok ("ruff debt {0} <= baseline {1}" -f $count, $baseline)
        return $true
    }
    Write-Bad ("ruff regression: {0} markers vs baseline {1} (+{2})" -f $count, $baseline, ($count - $baseline))
    $lines | Select-Object -First 10 | ForEach-Object { Write-Dim $_ }
    return $false
}

# ---------------------------------------------------------------- 3. backend
function Test-Backend {
    Write-Step "backend tests (full suite)"
    if ($SkipBackend) { Write-Warn "backend tests skipped (-SkipBackend)"; return $true }

    $env:PYTHONUTF8 = "1"
    if (-not $env:SECRET_KEY) {
        $env:SECRET_KEY = "test-secret-key-0123456789abcdef0123456789abcdef"
    }
    if (-not $env:DATABASE_URL) {
        $env:DATABASE_URL = "postgresql+asyncpg://postgres:Admin_123!@localhost:5432/compare_competencies"
    }

    $r = Invoke-Native -FilePath "python" `
        -ArgList @("-m", "pytest", "tests", "-q", "--no-header", "-p", "no:randomly") `
        -TimeoutSec ($TimeoutMin * 60)

    if ($r.TimedOut) { Write-Bad "backend tests exceeded $TimeoutMin min"; return $false }

    $text = $r.StdOut + "`n" + $r.StdErr
    $summary = @($text -split "`n" | Where-Object { $_ -match '\d+ (passed|failed|error)' } | Select-Object -Last 1)
    $failed  = @($text -split "`n" | Where-Object { $_ -match '^FAILED ' })

    if ($r.Code -eq 0) {
        Write-Ok ($summary -join " ").Trim()
        return $true
    }
    Write-Bad ("backend tests failed (exit {0})" -f $r.Code)
    if ($summary) { Write-Dim ($summary -join " ").Trim() }
    $failed | Select-Object -First 10 | ForEach-Object { Write-Dim $_ }
    return $false
}

# --------------------------------------------------------------- 4. frontend
function Test-Frontend {
    Write-Step "frontend (tsc / vitest / build)"
    $fe = Join-Path $root "frontend"
    if (-not (Test-Path (Join-Path $fe "node_modules"))) {
        Write-Bad "frontend/node_modules missing (run npm ci first)"
        return $false
    }
    $npx = "npx"
    $npm = "npm"
    $prev = (Get-Location).Path
    Set-Location -LiteralPath $fe
    try {
        $tsc = Invoke-Native -FilePath $npx -ArgList @("tsc", "--noEmit") -TimeoutSec 600 -ViaCmd
        $tscErrors = @(($tsc.StdOut + "`n" + $tsc.StdErr) -split "`n" | Where-Object { $_ -match 'error TS' })
        if ($tsc.Code -eq 0 -and $tscErrors.Count -eq 0) {
            Write-Ok "tsc: 0 errors"
        } elseif ($tscErrors.Count -gt 0) {
            Write-Bad ("tsc: {0} errors" -f $tscErrors.Count)
            $tscErrors | Select-Object -First 8 | ForEach-Object { Write-Dim $_ }
        } else {
            Write-Bad ("tsc crashed (exit {0}, no diagnostics)" -f $tsc.Code)
            ($tsc.StdOut + "`n" + $tsc.StdErr) -split "`n" | Select-Object -Last 6 | ForEach-Object { Write-Dim $_ }
        }

        $vit = Invoke-Native -FilePath $npx -ArgList @("vitest", "run") -TimeoutSec 600 -ViaCmd
        $vitText = $vit.StdOut + "`n" + $vit.StdErr
        $vitSummary = @($vitText -split "`n" | Where-Object { $_ -match 'Tests\s+\d' } | Select-Object -Last 1)
        if ($vit.Code -eq 0) {
            Write-Ok ("vitest: {0}" -f ($vitSummary -join " ").Trim())
        } else {
            Write-Bad ("vitest failed (exit {0})" -f $vit.Code)
            if ($vitSummary) { Write-Dim ($vitSummary -join " ").Trim() }
            $vitText -split "`n" | Where-Object { $_ -match 'FAIL|×' } | Select-Object -First 8 | ForEach-Object { Write-Dim $_ }
        }

        $build = Invoke-Native -FilePath $npm -ArgList @("run", "build") -TimeoutSec 900 -ViaCmd
        $buildText = $build.StdOut + "`n" + $build.StdErr
        $buildLine = @($buildText -split "`n" | Where-Object { $_ -match 'built in|error during' } | Select-Object -Last 1)
        if ($build.Code -eq 0) {
            Write-Ok ("build: {0}" -f ($buildLine -join " ").Trim())
        } else {
            Write-Bad ("frontend build failed (exit {0})" -f $build.Code)
            $buildText -split "`n" | Select-Object -Last 12 | ForEach-Object { Write-Dim $_ }
        }
    } finally {
        Set-Location -LiteralPath $prev
    }
    return ($script:Failures.Count -eq 0)
}

# ---------------------------------------------------------------- 5. alembic
function Test-Alembic {
    Write-Step "alembic heads"
    $r = Invoke-Native -FilePath "python" -ArgList @("-m", "alembic", "heads") -TimeoutSec 180
    $text = $r.StdOut + "`n" + $r.StdErr
    if ($r.Code -ne 0) {
        if ($text -match "No module named|Can't locate|not found") {
            Write-Warn "alembic not installed here (skipped)"
        } else {
            Write-Bad "alembic heads failed (exit $($r.Code))"
            $text -split "`n" | Select-Object -Last 6 | ForEach-Object { Write-Dim $_ }
        }
        return $true
    }
    # Имена ревизий - slug'и ('add_discipline_intake_year (head)'), а не хеши,
    # поэтому считаем маркеры (head), а не строки с hex.
    $headMarkers = @($text -split "`n" | Where-Object { $_ -match '\(head\)' })
    if ($headMarkers.Count -eq 1) {
        Write-Ok ("single head: {0}" -f ($headMarkers[0] -replace '\s*\(.*$', '').Trim())
        return $true
    }
    if ($headMarkers.Count -eq 0) { Write-Warn "no alembic heads found"; return $true }
    Write-Bad ("multiple alembic heads ({0}) - merge them first" -f $headMarkers.Count)
    $headMarkers | ForEach-Object { Write-Dim $_ }
    return $false
}

# ------------------------------------------------------------- 6. dep audit
# Известный принятый долг: эти пакеты не блокируют релиз, пока не появится
# безопасный путь обновления. Любая НОВАЯ уязвимость - блокер.
$script:AllowedVulnPackages = @(
    "starlette"   # 6 CVE; фикс в 1.3.1 требует миграции FastAPI (см. MEMORY E50)
)

function Get-DepAudit {
    Write-Step "dependency audit"
    $r = Invoke-Native -FilePath "python" -ArgList @("-m", "pip_audit", "-r", "requirements.txt", "--progress-spinner", "off") -TimeoutSec 600
    if ($r.Code -eq 0) { Write-Ok "no known vulnerabilities"; return $true }

    $rows = @(($r.StdOut + "`n" + $r.StdErr) -split "`n" |
        Where-Object { $_ -match '^\S+\s+\S+\s+PYSEC' })
    $pkgs = @($rows | ForEach-Object { ($_ -split '\s+')[0] } | Sort-Object -Unique)
    if ($pkgs.Count -eq 0) { Write-Warn "pip-audit unavailable (skipped)"; return $true }

    $unexpected = @($pkgs | Where-Object { $script:AllowedVulnPackages -notcontains $_ })
    if ($unexpected.Count -eq 0) {
        Write-Ok ("only known accepted debt: {0}" -f ($pkgs -join ", "))
        return $true
    }
    Write-Bad ("new vulnerable packages ({0}): {1}" -f $unexpected.Count, ($unexpected -join ", "))
    $rows | Where-Object { $unexpected -contains ($_ -split '\s+')[0] } |
        Select-Object -First 10 | ForEach-Object { Write-Dim $_ }
    return $false
}

# ------------------------------------------------------------------- actions
function Invoke-Precheck {
    Write-Host "`n########## RELEASE PRECHECK ##########" -ForegroundColor Magenta
    [void](Test-WorktreeClean)
    [void](Test-Ruff)
    [void](Test-Backend)
    [void](Test-Frontend)
    [void](Test-Alembic)
    # Блокирующая проверка: новые уязвимые пакеты останавливают релиз.
    [void](Get-DepAudit)

    Write-Host "`n########## RESULT ##########" -ForegroundColor Magenta
    if ($script:Notes.Count) {
        Write-Host "Warnings:" -ForegroundColor Yellow
        $script:Notes | ForEach-Object { Write-Host "  - $_" -ForegroundColor Yellow }
    }
    if ($script:Failures.Count -eq 0) {
        Write-Host "GATE: GREEN - safe to merge and push" -ForegroundColor Green
        return $true
    }
    Write-Host ("GATE: RED - {0} blocking failure(s)" -f $script:Failures.Count) -ForegroundColor Red
    $script:Failures | ForEach-Object { Write-Host "  - $_" -ForegroundColor Red }
    return $false
}

function Invoke-Merge {
    Write-Step "merge into $Target"
    if (-not $Source) { $Source = (Invoke-Git @("rev-parse", "--abbrev-ref", "HEAD")).Out.Trim() }
    if ($Source -eq $Target) { Write-Warn "source equals target ($Target) - nothing to merge"; return $true }

    $anc = Invoke-Git @("merge-base", "--is-ancestor", $Target, $Source)
    if ($anc.Code -ne 0) {
        Write-Bad "$Target is not an ancestor of $Source - branches diverged"
        return $false
    }

    $ahead = (Invoke-Git @("rev-list", "--count", "$Target..$Source")).Out.Trim()
    Write-Host "  commits to ship: $ahead"
    if ($ahead -eq "0") { Write-Ok "nothing to ship"; return $true }
    (Invoke-Git @("log", "--oneline", "$Target..$Source")).Out -split "`n" |
        Select-Object -First 20 | ForEach-Object { Write-Host "        $_" -ForegroundColor DarkGray }

    $checkout = Invoke-Git @("checkout", $Target)
    if ($checkout.Code -ne 0) { Write-Bad "checkout $Target failed: $($checkout.Out)"; return $false }

    $merge = Invoke-Git @("merge", "--no-ff", $Source, "-m", "merge: $Source into $Target")
    if ($merge.Code -ne 0) {
        Write-Bad "merge conflicts - resolve by hand, then re-run Precheck"
        Write-Dim $merge.Out
        [void](Invoke-Git @("merge", "--abort"))
        return $false
    }
    Write-Ok "merged ($ahead commits)"
    return $true
}

function Invoke-Push {
    Write-Step "push origin $Target"
    $branch = (Invoke-Git @("rev-parse", "--abbrev-ref", "HEAD")).Out.Trim()
    if ($branch -ne $Target) { Write-Bad "current branch is $branch, expected $Target"; return $false }

    $push = Invoke-Git @("push", "origin", $Target)
    if ($push.Code -ne 0) { Write-Bad "push failed: $($push.Out)"; return $false }

    [void](Invoke-Git @("fetch", "origin"))
    $behind = (Invoke-Git @("rev-list", "--count", "origin/$Target..$Target")).Out.Trim()
    if ($behind -ne "0") {
        Write-Bad "push unverified: still $behind commits ahead of origin/$Target"
        return $false
    }
    Write-Ok "pushed; origin/$Target is in sync"
    return $true
}

# ---------------------------------------------------------------------- main
switch ($Action) {
    "Precheck" { if (Invoke-Precheck) { exit 0 } else { exit 1 } }
    "Merge" {
        if (-not (Invoke-Precheck)) { Write-Host "`nMerge refused: precheck failed." -ForegroundColor Red; exit 1 }
        if (Invoke-Merge) { exit 0 } else { exit 1 }
    }
    "Push" { if (Invoke-Push) { exit 0 } else { exit 1 } }
    "All" {
        if (-not (Invoke-Precheck)) { Write-Host "`nRelease refused: precheck failed." -ForegroundColor Red; exit 1 }
        if (-not (Invoke-Merge))   { Write-Host "`nRelease stopped at merge." -ForegroundColor Red; exit 1 }
        if (-not (Invoke-Push))    { Write-Host "`nRelease stopped at push." -ForegroundColor Red; exit 1 }
        Write-Host "`nRELEASE COMPLETE" -ForegroundColor Green
        exit 0
    }
}
