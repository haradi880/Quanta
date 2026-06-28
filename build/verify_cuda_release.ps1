param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$DistributionPath = ".\dist\Quanta"
)

$ErrorActionPreference = "Stop"
$DistributionPath = (Resolve-Path -LiteralPath $DistributionPath).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$Executable = Join-Path $DistributionPath "Quanta.exe"

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Quanta.exe is missing from distribution: $DistributionPath"
}
$NvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $NvidiaSmi) {
    throw "CUDA acceptance requires an NVIDIA driver and nvidia-smi."
}
$GpuInventory = & $NvidiaSmi.Source --query-gpu=name,driver_version,memory.total `
    --format=csv,noheader
if ($LASTEXITCODE -ne 0 -or -not $GpuInventory) {
    throw "nvidia-smi returned no healthy GPU inventory."
}

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HARADIBOTS_CACHE_ROOT = Join-Path $env:TEMP (
    "quanta-cuda-acceptance-" + [guid]::NewGuid().ToString("N")
)

$DoctorText = (& $Executable doctor --json 2>&1) -join [Environment]::NewLine
if ($LASTEXITCODE -ne 0) {
    throw "Packaged doctor failed: $DoctorText"
}
$Doctor = $DoctorText | ConvertFrom-Json
$CompletionPath = [string]$Doctor.offline_native_assets.'llama-completion'.path
if ($CompletionPath -notmatch '[\\/]vendor[\\/]cuda[\\/]llama-completion\.exe$') {
    throw "Quanta selected CPU llama.cpp despite an available NVIDIA GPU."
}

$JobText = (& $Executable run --model $ModelPath --json 2>&1) -join `
    [Environment]::NewLine
if ($LASTEXITCODE -ne 0) {
    throw "Packaged CUDA job failed: $JobText"
}
if ($JobText -notmatch '"backend":\s*"llama\.cpp CUDA"') {
    throw "Planner did not select the llama.cpp CUDA backend."
}
if (
    $JobText -notmatch '(?i)offloaded\s+\d+/\d+\s+layers\s+to\s+GPU' -and
    $JobText -notmatch '(?i)CUDA0'
) {
    throw "Inference output contains no evidence of CUDA layer offload."
}
if ($JobText -notmatch '"event_type":\s*"teardown_complete"') {
    throw "CUDA job emitted no teardown_complete event."
}
if ($JobText -notmatch '"forced_kill_count":\s*0') {
    throw "CUDA job required a forced worker kill."
}
if (
    $JobText -notmatch
    '(?s)"event_type":\s*"complete".*?"state":\s*"IDLE".*?"status":\s*"complete"'
) {
    throw "CUDA job did not finish at IDLE/complete."
}

[ordered]@{
    status = "passed"
    product = "HaradiBots Quanta"
    executable = $Executable
    model = $ModelPath
    gpu_inventory = @($GpuInventory)
    cuda_binary = $CompletionPath
    cuda_backend = $true
    layer_offload_observed = $true
    teardown_complete = $true
    forced_kill_count = 0
    final_state = "IDLE"
} | ConvertTo-Json -Depth 4
