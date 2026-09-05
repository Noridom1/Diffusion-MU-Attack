<#!
.SYNOPSIS
    Run the isolated timestep-EOT attack on the three baseline failures.

This is a local PowerShell runner.  It never writes to the baseline result
folder; EOT logs are placed under a separate ``unlearndiff_eot`` root.
#>

[CmdletBinding()]
param(
    [int[]]$Indices = @(7, 12, 19),
    [string]$Bundle = "files/downloads/nudity_n20_sample2024_run0_eval_bundle",
    [string]$Experiment = "nudity_n20_sample2024_run0",
    [string]$OutputRoot = "files/results/nudity_n20_sample2024_run0/unlearndiff_eot",
    [string]$Checkpoint = "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
    [string]$CachePath = ".cache",
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$config = Join-Path $repoRoot "configs/nudity/text_grad_eot_esd_nudity_classifier.json"
$dataset = Join-Path $repoRoot (Join-Path $Bundle (Join-Path "files/dataset" $Experiment))
$output = Join-Path $repoRoot $OutputRoot

if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Missing EOT config: $config"
}
if (-not (Test-Path -LiteralPath $dataset -PathType Container)) {
    throw "Missing dataset: $dataset"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Missing target checkpoint: $Checkpoint (pass -Checkpoint if it is stored elsewhere)"
}

New-Item -ItemType Directory -Path $output -Force | Out-Null

foreach ($idx in $Indices) {
    $runDir = Join-Path $output "attack_idx_$idx"
    $done = Join-Path $runDir ".done"
    if (Test-Path -LiteralPath $done -PathType Leaf) {
        Write-Output "[EOT] skip completed index $idx"
        continue
    }
    if (Test-Path -LiteralPath $runDir -PathType Container) {
        throw "Incomplete result directory already exists: $runDir. Move it aside before rerunning."
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Write-Output "[EOT] starting index $idx"
    & $Python "src/execs/attack.py" `
        "--config-file" $config `
        "--task.target_ckpt" $Checkpoint `
        "--task.cache_path" $CachePath `
        "--task.dataset_path" $dataset `
        "--attacker.attack_idx" $idx `
        "--logger.name" "attack_idx_$idx" `
        "--logger.json.root" $output 2>&1 | Tee-Object -FilePath (Join-Path $runDir "process.log")
    if ($LASTEXITCODE -ne 0) {
        throw "EOT attack failed for index $idx (exit code $LASTEXITCODE)"
    }
    New-Item -ItemType File -Path $done | Out-Null
    Write-Output "[EOT] completed index $idx"
}

Write-Output "EOT results: $output"
