# Run the W&B tuning sweeps for logistic regression and then the random forest at every resolution.
# Requires the ml_env_1 conda environment. Run from any folder: .\scripts\tune_classical.ps1

$Configs = @("configs/classical_logreg.yaml", "configs/classical_rf.yaml")

if ($env:CONDA_DEFAULT_ENV -ne "ml_env_1") {
    $Message = "Activate the ml_env_1 conda environment first (active: '$env:CONDA_DEFAULT_ENV')."
    Write-Host $Message -ForegroundColor Red
    exit 1
}

Push-Location (Split-Path -Parent $PSScriptRoot)
$ExitCode = 0
foreach ($Config in $Configs) {
    Write-Host "Tuning with $Config"
    python src/tune_classical.py --config $Config
    if ($LASTEXITCODE -ne 0) {
        $ExitCode = $LASTEXITCODE
        Write-Host "Tuning failed for $Config (exit code $ExitCode)." -ForegroundColor Red
        break
    }
}
Pop-Location
exit $ExitCode
