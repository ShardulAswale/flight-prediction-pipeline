$ErrorActionPreference = "Stop"

$ProjectRoot = "C:\Code\flight-disruption-prediction"
$Python = "C:\Code\.venv311\Scripts\python.exe"
$LogDir = Join-Path $ProjectRoot "logs"

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "data\raw\opensky") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $ProjectRoot "data\processed") | Out-Null

Set-Location $ProjectRoot

Write-Host "Using Python:" $Python
& $Python --version

Write-Host "Skipping D: data copy. This run will redownload/rebuild ADS-B on C:."

$SampleDates = @(
    "2022-01-03", "2022-01-10", "2022-01-17", "2022-01-24", "2022-01-31",
    "2022-02-07", "2022-02-14", "2022-02-21", "2022-02-28",
    "2022-03-07", "2022-03-14", "2022-03-21", "2022-03-28",
    "2022-04-04", "2022-04-11", "2022-04-18", "2022-04-25",
    "2022-05-02", "2022-05-09", "2022-05-16", "2022-05-23", "2022-05-30",
    "2022-06-06", "2022-06-13", "2022-06-20", "2022-06-27"
)

$DateArgs = @()
foreach ($d in $SampleDates) {
    $DateArgs += "--sample-date"
    $DateArgs += $d
}

$FetchArgs = @(
    "scripts\fetch_opensky_samples.py",
    "--hour-start", "0",
    "--hour-end", "23",
    "--max-workers", "8",
    "--archive-dir", "data\raw\opensky",
    "--single-output-file", "data\processed\adsb_combined.parquet"
) + $DateArgs

$CombinedOutput = Join-Path $ProjectRoot "data\processed\adsb_combined.parquet"
if (Test-Path $CombinedOutput) { Remove-Item $CombinedOutput -Force }

Write-Host "Starting six-month OpenSky ADS-B build: Jan-Jun 2022 weekly samples"
& $Python @FetchArgs
if ($LASTEXITCODE -ne 0) { throw "OpenSky fetch failed with exit code $LASTEXITCODE" }

Write-Host "Starting full pipeline: ingest -> features -> merge -> weather -> label -> validate -> train -> drift"
& $Python main.py --stage all --force
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }

Write-Host "Six-month pipeline completed."



