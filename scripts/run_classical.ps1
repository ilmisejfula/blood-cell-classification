# Run the final classical runs (best tuned settings, 5 seeds, one test evaluation) with W&B logging.
# Requires the ml_env_1 conda environment. Run from any folder: .\scripts\run_classical.ps1

if ($env:CONDA_DEFAULT_ENV -ne "ml_env_1") {
    $Message = "Activate the ml_env_1 conda environment first (active: '$env:CONDA_DEFAULT_ENV')."
    Write-Host $Message -ForegroundColor Red
    exit 1
}

Push-Location (Split-Path -Parent $PSScriptRoot)
python src/run_classical.py --wandb
$ExitCode = $LASTEXITCODE
if ($ExitCode -ne 0) {
    Write-Host "Final runs failed (exit code $ExitCode)." -ForegroundColor Red
}
Pop-Location
exit $ExitCode
