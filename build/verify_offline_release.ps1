param(
    [Parameter(Mandatory = $true)]
    [string]$ModelPath,
    [string]$DistributionPath = ".\dist\Quanta",
    [string]$ExpectedExeSha256 = ""
)

$ErrorActionPreference = "Stop"
$DistributionPath = (Resolve-Path -LiteralPath $DistributionPath).Path
$ModelPath = (Resolve-Path -LiteralPath $ModelPath).Path
$Executable = Join-Path $DistributionPath "Quanta.exe"

if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "Quanta.exe is missing from distribution: $DistributionPath"
}
if ([IO.Path]::GetExtension($ModelPath).ToLowerInvariant() -ne ".gguf") {
    throw "Offline acceptance requires a local .gguf model."
}
if ($ExpectedExeSha256) {
    $ActualHash = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    if ($ActualHash -ne $ExpectedExeSha256.ToUpperInvariant()) {
        throw "Quanta.exe SHA-256 mismatch."
    }
}

# Any accidental Hugging Face/Transformers network path must fail immediately.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:HARADIBOTS_CACHE_ROOT = Join-Path $env:TEMP (
    "quanta-offline-acceptance-" + [guid]::NewGuid().ToString("N")
)

$DoctorOutput = & $Executable doctor --json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Packaged doctor failed: $($DoctorOutput -join [Environment]::NewLine)"
}
$Doctor = ($DoctorOutput -join [Environment]::NewLine) | ConvertFrom-Json
if ($Doctor.status -ne "healthy" -or -not $Doctor.redis_stopped) {
    throw "Packaged doctor did not report a healthy, stopped runtime."
}
foreach ($Command in @("PING", "HSET", "HGETALL", "SCAN")) {
    if (-not $Doctor.resp_checks.PSObject.Properties.Name.Contains($Command)) {
        throw "Packaged doctor omitted RESP check: $Command"
    }
}

$JobOutput = & $Executable run --model $ModelPath --json 2>&1
if ($LASTEXITCODE -ne 0) {
    throw "Packaged local-GGUF job failed: $($JobOutput -join [Environment]::NewLine)"
}
$Text = $JobOutput -join [Environment]::NewLine
if ($Text -notmatch '"event_type":\s*"teardown_complete"') {
    throw "Packaged job emitted no teardown_complete event."
}
if ($Text -notmatch '"forced_kill_count":\s*0') {
    throw "Packaged job required a forced worker kill."
}
if (
    $Text -notmatch
    '(?s)"event_type":\s*"complete".*?"state":\s*"IDLE".*?"status":\s*"complete"'
) {
    throw "Packaged job did not finish at IDLE/complete."
}

$Remaining = Get-Process -ErrorAction SilentlyContinue |
    Where-Object {
        $_.ProcessName -in @(
            "Quanta",
            "llama-completion",
            "llama-quantize",
            "llama-perplexity",
            "GarnetServer"
        )
    }
if ($Remaining) {
    throw "Release left native processes alive: $($Remaining.ProcessName -join ', ')"
}

[ordered]@{
    status = "passed"
    product = "HaradiBots Quanta"
    executable = $Executable
    executable_sha256 = (Get-FileHash -LiteralPath $Executable -Algorithm SHA256).Hash
    model = $ModelPath
    offline_flags = $true
    doctor = "healthy"
    teardown_complete = $true
    forced_kill_count = 0
    final_state = "IDLE"
} | ConvertTo-Json -Depth 4
