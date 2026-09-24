# Phase-2 ML correction - ONE command, resumable (finished jobs are cached; re-run the same command to continue).
#   powershell -ExecutionPolicy Bypass -File C:\Mine\Drone\scripts\run_ml_correction_phase2.ps1
# Runs in the existing 'drone-ml' conda env (XGBoost / CatBoost / LightGBM / PyTorch + RTX 3070).
# Development only: MH_05 is never built, loaded or evaluated by this script.
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Res = Join-Path $Root "ml\correction_phase2\results"; New-Item -ItemType Directory -Force -Path $Res | Out-Null
$Console = Join-Path $Res "runner_console.txt"
function Say($m) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') [runner] $m"; Write-Host $l -ForegroundColor Cyan; Add-Content -Path $Console -Value $l -Encoding UTF8 }

# ---- locate conda / the drone-ml environment (no installs, no system changes)
$CondaCands = @(@($env:CONDA_EXE, "C:\ProgramData\anaconda3\Scripts\conda.exe", "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
                "$env:USERPROFILE\miniconda3\Scripts\conda.exe", "C:\ProgramData\miniconda3\Scripts\conda.exe") | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique)
$EnvCands = @(@("$env:USERPROFILE\.conda\envs\drone-ml", "C:\ProgramData\anaconda3\envs\drone-ml", "$env:USERPROFILE\anaconda3\envs\drone-ml",
              "$env:USERPROFILE\miniconda3\envs\drone-ml") | Where-Object { Test-Path (Join-Path $_ "python.exe") })
$env:PYTHONUNBUFFERED = "1"; $env:PYTHONIOENCODING = "utf-8"; $env:KMP_DUPLICATE_LIB_OK = "TRUE"
$Script = Join-Path $Root "ml\correction_phase2\src\p2_run.py"
Say "root $Root | conda: $($CondaCands -join ', ') | env: $($EnvCands -join ', ')"
if ($EnvCands.Count -gt 0) {
    $EnvDir = [string]$EnvCands[0]; $global:LASTEXITCODE = 99
    # activate-equivalent PATH for the env (DLLs for numpy/torch), then run the env's python directly
    $env:PATH = "$EnvDir;$EnvDir\Library\mingw-w64\bin;$EnvDir\Library\usr\bin;$EnvDir\Library\bin;$EnvDir\Scripts;$EnvDir\bin;" + $env:PATH
    $env:CONDA_PREFIX = $EnvDir
    Say "running $EnvDir\python.exe $Script"
    & "$EnvDir\python.exe" -u $Script @args 2>&1 | ForEach-Object { "$_" } | Tee-Object -Variable out | ForEach-Object { Write-Host $_; Add-Content -Path $Console -Value $_ -Encoding UTF8 }
    $code = $LASTEXITCODE
} elseif ($CondaCands.Count -gt 0) {
    Say "running via conda run -n drone-ml"
    & $CondaCands[0] run --no-capture-output -n drone-ml python -u $Script @args 2>&1 | ForEach-Object { "$_" } | ForEach-Object { Write-Host $_; Add-Content -Path $Console -Value $_ -Encoding UTF8 }
    $code = $LASTEXITCODE
} else { Say "FATAL: drone-ml environment not found"; $code = 9 }
Say "exit code $code"
Set-Content -Path (Join-Path $Res "RUNNER_EXIT.txt") -Value "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exit=$code" -Encoding UTF8
exit $code
