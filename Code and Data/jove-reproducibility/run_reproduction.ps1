param(
    [string]$PythonExecutable = "python",
    [int]$Bootstrap = 2000
)
$ErrorActionPreference = "Stop"
$runFolder = Join-Path $PSScriptRoot ("results_" + (Get-Date -Format "yyyyMMdd_HHmmss_fff"))
& $PythonExecutable (Join-Path $PSScriptRoot "reproduce.py") --bootstrap $Bootstrap --output $runFolder
if ($LASTEXITCODE -ne 0) { throw "Reproduction failed. Review the error and run_manifest.json." }
Write-Host "Results saved to: $runFolder"
