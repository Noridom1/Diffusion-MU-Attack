<#!
.SYNOPSIS
    Run the baseline-timestep HotFlip attack on baseline failures only.

The default index set is intentionally limited to 7, 12, and 19, which are
the three samples that failed the 20-sample baseline attack. Results are
written under a separate HotFlip root.
#>

[CmdletBinding()]
param(
    [int[]]$Indices = @(7, 12, 19),
    [string]$Bundle = "files/downloads/nudity_n20_sample2024_run0_eval_bundle",
    [string]$Experiment = "nudity_n20_sample2024_run0",
    [string]$OutputRoot = "files/results/nudity_n20_sample2024_run0/unlearndiff_hotflip",
    [string]$Checkpoint = "files/pretrained/SD-1-4/ESD_ckpt/Nudity-ESDx1-UNET-SD.pt",
    [string]$CachePath = ".cache",
    [string]$Python = "python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$config = Join-Path $repoRoot "configs/nudity/text_grad_hotflip_esd_nudity_classifier.json"
$dataset = Join-Path $repoRoot (Join-Path $Bundle (Join-Path "files/dataset" $Experiment))
$output = Join-Path $repoRoot $OutputRoot

if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Missing HotFlip config: $config"
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
        Write-Output "[HotFlip] skip completed index $idx"
        continue
    }
    if (Test-Path -LiteralPath $runDir -PathType Container) {
        throw "Incomplete result directory already exists: $runDir. Move it aside before rerunning."
    }

    New-Item -ItemType Directory -Path $runDir -Force | Out-Null
    Write-Output "[HotFlip] starting index $idx"
    & $Python "src/execs/attack.py" `
        "--config-file" $config `
        "--task.target_ckpt" $Checkpoint `
        "--task.cache_path" $CachePath `
        "--task.dataset_path" $dataset `
        "--attacker.attack_idx" $idx `
        "--logger.name" "attack_idx_$idx" `
        "--logger.json.root" $output 2>&1 | Tee-Object -FilePath (Join-Path $runDir "process.log")
    if ($LASTEXITCODE -ne 0) {
        throw "HotFlip attack failed for index $idx (exit code $LASTEXITCODE)"
    }
    New-Item -ItemType File -Path $done | Out-Null
    Write-Output "[HotFlip] completed index $idx"
}

Write-Output "HotFlip results: $output"
