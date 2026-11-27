# Run the W&B tuning sweep for one CNN at one resolution.
# Requires the ml_env_1 conda environment. Run from any folder:
#   .\scripts\tune_cnn.ps1 -Model SmallCNN -Resolution 28 [-Count N]

param(
    [Parameter(Mandatory)][string]$Model,
    [Parameter(Mandatory)][int]$Resolution,
    [int]$Count
)

if ($env:CONDA_DEFAULT_ENV -ne "ml_env_1") {
    $Message = "Activate the ml_env_1 conda environment first (active: '$env:CONDA_DEFAULT_ENV')."
    Write-Host $Message -ForegroundColor Red
    exit 1
}

$Arguments = @("--model", $Model, "--resolution", $Resolution)
if ($PSBoundParameters.ContainsKey("Count")) {
    $Arguments += @("--count", $Count)
}

Push-Location (Split-Path -Parent $PSScriptRoot)
Write-Host "Tuning $Model at $Resolution px"
python src/tune_cnn.py @Arguments
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    Write-Host "Tuning failed for $Model at $Resolution px (exit code $ExitCode)." -ForegroundColor Red
}
Pop-Location
exit $ExitCode
