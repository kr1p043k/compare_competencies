#Requires -Version 5.1
<#
.SYNOPSIS
    Verifies the newest compare_competencies database backups.

.DESCRIPTION
    Finds the newest backup of BOTH kinds produced by this repo:
      * custom-format pg_dump:  data\backups\backup_<UTC>.sql   (restorable with pg_restore)
      * plain SQL (non pg_restore-able): backups\db_<date>.sql  (restorable with psql -f)

    For each backup it:
      1. checks readability (pg_restore --list for custom; psql restore for plain),
      2. restores into a throwaway scratch database (created fresh),
      3. runs SELECT 1 and SELECT COUNT(*) FROM users,
      4. drops the scratch database,
      5. prints PASS/FAIL per backup and an overall result.

    It runs PostgreSQL client tools inside the running container (default
    competency-postgres). The production database name is refused as a target.

.PARAMETER DryRun
    Print the plan (files found, scratch DB, commands) and exit 0 without touching anything.

.PARAMETER BackupPath
    Verify exactly this file instead of auto-discovering the newest ones.
    Kind (custom/plain) is auto-detected.

.PARAMETER ProdDb
    Production database name (default compare_competencies). Never used as a target.

.PARAMETER Container
    PostgreSQL container name (default competency-postgres).

.PARAMETER DbUser
    PostgreSQL user for the client tools (default postgres).

.EXAMPLE
    powershell -File scripts/verify_backup.ps1 -DryRun

.EXAMPLE
    powershell -File scripts/verify_backup.ps1

.EXAMPLE
    powershell -File scripts/verify_backup.ps1 -BackupPath data\backups\backup_20260101_000000.sql
#>
[CmdletBinding()]
param(
    [switch]$DryRun,
    [string]$BackupPath,
    [string]$ProdDb = 'compare_competencies',
    [string]$ScratchDb,
    [string]$Container = 'competency-postgres',
    [string]$DbUser = 'postgres'
)

$ErrorActionPreference = 'Stop'

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$CustomDir  = Join-Path $RepoRoot 'data\backups'   # pg_dump --format=custom
$PlainDir   = Join-Path $RepoRoot 'backups'        # plain SQL: db_<date>.sql

$script:Failures = 0
$script:Checked  = 0

function Write-Info { param([string]$m) Write-Host "[verify-backup] $m" }
function Write-Ok   { param([string]$m) Write-Host "[ PASS ] $m" -ForegroundColor Green }
function Write-Bad  { param([string]$m) Write-Host "[ FAIL ] $m" -ForegroundColor Red }
function Write-Warn { param([string]$m) Write-Host "[ WARN ] $m" -ForegroundColor Yellow }

function Get-NewestFile {
    param([string]$Dir, [string]$Filter)
    if (-not (Test-Path -LiteralPath $Dir)) { return $null }
    Get-ChildItem -LiteralPath $Dir -Filter $Filter -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

# Runs `docker <args...>` and returns output + exit code without throwing on stderr.
function Invoke-Docker {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$DockerArgs)
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $out  = & docker @DockerArgs 2>&1
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    return [pscustomobject]@{
        Output   = (($out | Out-String).Trim())
        ExitCode = $code
    }
}

function Assert-NotProd {
    param([string]$Name)
    if ($Name -eq $ProdDb) {
        throw "Refusing to target the production database '$ProdDb'."
    }
}

# --- discover backups -------------------------------------------------------
$targets = New-Object System.Collections.Generic.List[object]

if ($BackupPath) {
    if (-not (Test-Path -LiteralPath $BackupPath)) {
        throw "BackupPath not found: $BackupPath"
    }
    $item = Get-Item -LiteralPath $BackupPath
    $kind = if ($item.Name -like 'backup_*') { 'custom' } else { 'plain' }
    $targets.Add([pscustomobject]@{ Kind = $kind; File = $item })
}
else {
    $custom = Get-NewestFile -Dir $CustomDir -Filter 'backup_*.sql'
    $plain  = Get-NewestFile -Dir $PlainDir  -Filter 'db_*.sql'
    if ($custom) { $targets.Add([pscustomobject]@{ Kind = 'custom'; File = $custom }) }
    if ($plain)  { $targets.Add([pscustomobject]@{ Kind = 'plain';  File = $plain  }) }
    if (-not $custom) { Write-Warn "No custom-format backup found in $CustomDir (backup_*.sql)" }
    if (-not $plain)  { Write-Warn "No plain-SQL backup found in $PlainDir (db_*.sql)" }
}

$scratch = if ($ScratchDb) { $ScratchDb } else {
    'cc_verify_' + (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + (Get-Random -Maximum 99999)
}
Assert-NotProd -Name $scratch

# --- plan / dry run ---------------------------------------------------------
Write-Host ''
Write-Info "repo root     : $RepoRoot"
Write-Info "container     : $Container (user $DbUser)"
Write-Info "production db : $ProdDb (refused as target)"
Write-Info "scratch db    : $scratch"
Write-Info "backups found : $($targets.Count)"
foreach ($t in $targets) {
    Write-Info ("  - [{0,-6}] {1} ({2:N1} MB)" -f $t.Kind, $t.File.FullName, ($t.File.Length / 1MB))
}
Write-Host ''

if ($DryRun) {
    Write-Info 'DRY RUN - no changes made. Planned actions:'
    Write-Info "  1. docker cp <backup> ${Container}:/tmp/cc_verify_backup.dump"
    Write-Info "  2. custom: docker exec $Container pg_restore --list /tmp/cc_verify_backup.dump"
    Write-Info "     plain : (verified by restoring below)"
    Write-Info "  3. docker exec $Container psql -U $DbUser -d postgres -c `"CREATE DATABASE $scratch`""
    Write-Info "  4. custom: docker exec $Container pg_restore --no-owner --no-privileges -U $DbUser -d $scratch /tmp/cc_verify_backup.dump"
    Write-Info "     plain : docker exec $Container psql -U $DbUser -d $scratch -f /tmp/cc_verify_backup.dump"
    Write-Info "  5. docker exec $Container psql -U $DbUser -d $scratch -tAc `"SELECT 1;`""
    Write-Info "  6. docker exec $Container psql -U $DbUser -d $scratch -tAc `"SELECT count(*) FROM users;`""
    Write-Info "  7. docker exec $Container psql -U $DbUser -d postgres -c `"DROP DATABASE $scratch`""
    Write-Info "  8. docker exec $Container rm -f /tmp/cc_verify_backup.dump"
    Write-Info 'PASS/FAIL would be printed per backup.'
    if ($targets.Count -eq 0) {
        Write-Warn "No backups found yet (looked in $CustomDir and $PlainDir)."
    }
    exit 0
}

if ($targets.Count -eq 0) {
    Write-Bad "No backups found to verify (looked in $CustomDir and $PlainDir)."
    exit 1
}

# --- docker available? ------------------------------------------------------
$dockerCheck = Invoke-Docker version --format '{{.Server.Version}}'
if ($dockerCheck.ExitCode -ne 0) {
    Write-Bad "docker is not available (is the engine running?). $($dockerCheck.Output)"
    exit 1
}

$tmpInContainer = "/tmp/cc_verify_backup_$([guid]::NewGuid().ToString('N')).dump"

foreach ($t in $targets) {
    $script:Checked++
    $label = "[$($t.Kind)] $($t.File.Name)"
    Write-Info "verifying $label"

    $ok = $true

    # 1) copy the dump into the container
    $cp = Invoke-Docker cp $t.File.FullName "${Container}:${tmpInContainer}"
    if ($cp.ExitCode -ne 0) {
        Write-Bad "$label - docker cp failed: $($cp.Output)"
        $script:Failures++
        continue
    }

    # 2) custom format: readability via pg_restore --list
    if ($t.Kind -eq 'custom') {
        $list = Invoke-Docker exec $Container pg_restore --list $tmpInContainer
        if ($list.ExitCode -ne 0) {
            Write-Bad "$label - pg_restore --list failed (dump not readable): $($list.Output)"
            $ok = $false
        }
        elseif ($list.Output -notmatch 'TABLE public users') {
            Write-Warn "$label - readable, but 'TABLE public users' not found in listing"
        }
        else {
            Write-Info "$label - pg_restore --list OK (users table present)"
        }
    }

    if ($ok) {
        # 3) create scratch database
        Invoke-Docker exec $Container psql -U $DbUser -d postgres -v ON_ERROR_STOP=1 `
            -c "DROP DATABASE IF EXISTS $scratch" | Out-Null
        $create = Invoke-Docker exec $Container psql -U $DbUser -d postgres -v ON_ERROR_STOP=1 `
            -c "CREATE DATABASE $scratch"
        if ($create.ExitCode -ne 0) {
            Write-Bad "$label - CREATE DATABASE $scratch failed: $($create.Output)"
            $ok = $false
        }
    }

    if ($ok) {
        # 4) restore into scratch
        if ($t.Kind -eq 'custom') {
            $restore = Invoke-Docker exec $Container pg_restore --no-owner --no-privileges `
                -U $DbUser -d $scratch $tmpInContainer
        }
        else {
            $restore = Invoke-Docker exec $Container psql -U $DbUser -d $scratch `
                -v ON_ERROR_STOP=1 -f $tmpInContainer
        }
        if ($restore.ExitCode -ne 0) {
            Write-Bad "$label - restore into $scratch failed: $($restore.Output)"
            $ok = $false
        }
    }

    $userCount = $null
    if ($ok) {
        # 5) SELECT 1
        $sel1 = Invoke-Docker exec $Container psql -U $DbUser -d $scratch -tAc 'SELECT 1;'
        if ($sel1.ExitCode -ne 0 -or ($sel1.Output.Trim()) -ne '1') {
            Write-Bad "$label - SELECT 1 failed: $($sel1.Output)"
            $ok = $false
        }
        else {
            # 6) SELECT COUNT(*) FROM users
            $cnt = Invoke-Docker exec $Container psql -U $DbUser -d $scratch -tAc 'SELECT count(*) FROM users;'
            if ($cnt.ExitCode -ne 0) {
                Write-Bad "$label - SELECT count(*) FROM users failed: $($cnt.Output)"
                $ok = $false
            }
            else {
                $userCount = $cnt.Output.Trim()
            }
        }
    }

    # 7) drop scratch + cleanup (always)
    Invoke-Docker exec $Container psql -U $DbUser -d postgres -v ON_ERROR_STOP=1 `
        -c "DROP DATABASE IF EXISTS $scratch" | Out-Null
    Invoke-Docker exec $Container rm -f $tmpInContainer | Out-Null

    if ($ok) {
        Write-Ok "$label - restored OK, users=$userCount"
    }
    else {
        Write-Bad "$label - verification FAILED"
        $script:Failures++
    }
}

Write-Host ''
if ($script:Failures -eq 0) {
    Write-Ok "ALL backups verified ($($script:Checked) checked)."
    exit 0
}

Write-Bad "$($script:Failures) of $($script:Checked) backup(s) FAILED."
exit 1
