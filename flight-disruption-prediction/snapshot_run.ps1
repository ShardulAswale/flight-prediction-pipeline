param(
    [Parameter(Mandatory = $true)]
    [string]$RunName
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$Destination = Join-Path $RepoRoot $RunName

if (Test-Path $Destination) {
    throw "Destination already exists: $Destination"
}

New-Item -ItemType Directory -Path $Destination | Out-Null

Write-Host "Creating snapshot:" $Destination

Copy-Item (Join-Path $RepoRoot "configs\config.yaml") (Join-Path $Destination "config.yaml") -Force
Copy-Item (Join-Path $RepoRoot "models") (Join-Path $Destination "models") -Recurse -Force
Copy-Item (Join-Path $RepoRoot "logs") (Join-Path $Destination "logs") -Recurse -Force
Copy-Item (Join-Path $RepoRoot "outputs") (Join-Path $Destination "outputs") -Recurse -Force

$ExcludedArtifacts = @(
    (Join-Path $Destination "outputs\shap_values.parquet"),
    (Join-Path $Destination "outputs\shap_force_plot_0.html"),
    (Join-Path $Destination "outputs\shap_force_plot_1.html"),
    (Join-Path $Destination "outputs\shap_force_plot_2.html")
)

$ExistingExcludedArtifacts = $ExcludedArtifacts | Where-Object { Test-Path $_ }
if ($ExistingExcludedArtifacts.Count -gt 0) {
    Remove-Item -LiteralPath $ExistingExcludedArtifacts -Force
}

Write-Host "Snapshot complete."
Write-Host "Saved to:" $Destination
