param(
    [int]$FeaturePid,
    [string]$ProjectRoot = "C:\Code\flight-disruption-prediction",
    [string]$Python = "C:\Code\.venv311\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
Set-Location $ProjectRoot
$LogDir = Join-Path $ProjectRoot "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$RunLog = Join-Path $LogDir "run_after_features_$Stamp.log"

function Log($Message) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | $Message"
    $line | Tee-Object -FilePath $RunLog -Append
}

function RunStep($Name, [string[]]$ArgsList) {
    Log "START $Name :: $Python $($ArgsList -join ' ')"
    & $Python @ArgsList 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
    Log "DONE $Name"
}

try {
    Log "Waiting for feature PID $FeaturePid"
    while (Get-Process -Id $FeaturePid -ErrorAction SilentlyContinue) {
        Start-Sleep -Seconds 60
    }
    Log "Feature PID $FeaturePid ended. Validating feature/sketch parquet outputs."

    $validateScript = @'
from pathlib import Path
import pyarrow.parquet as pq
for name in ["trajectory_features.parquet", "trajectory_sketches.parquet"]:
    path = Path("data/processed") / name
    if not path.exists():
        raise SystemExit(f"missing {path}")
    pf = pq.ParquetFile(path)
    print(f"{name}: rows={pf.metadata.num_rows:,}, columns={len(pf.schema.names)}, row_groups={pf.metadata.num_row_groups}")
'@
    $validateScript | & $Python - 2>&1 | Tee-Object -FilePath $RunLog -Append
    if ($LASTEXITCODE -ne 0) { throw "Feature output validation failed" }

    RunStep "fetch_weather_6month" @("scripts\fetch_weather.py", "--start", "2022-01-01", "--end", "2022-06-30", "--max-stations", "80", "--output", "data\raw\metar.parquet")
    RunStep "merge" @("main.py", "--stage", "merge", "--force")
    RunStep "weather" @("main.py", "--stage", "weather", "--force")
    RunStep "label" @("main.py", "--stage", "label", "--force")
    RunStep "validate" @("main.py", "--stage", "validate", "--force")
    RunStep "train" @("main.py", "--stage", "train", "--force")
    Log "ALL DONE"
} catch {
    Log "FAILED :: $($_.Exception.Message)"
    exit 1
}
