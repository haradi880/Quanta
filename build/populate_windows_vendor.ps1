param(
    [string]$Python = "python",
    [Parameter(Mandatory = $true)]
    [string]$RedisBinary,
    [Parameter(Mandatory = $true)]
    [string]$RedisRuntimeDirectory,
    [Parameter(Mandatory = $true)]
    [string]$RedisSource,
    [Parameter(Mandatory = $true)]
    [string]$RedisLicense,
    [Parameter(Mandatory = $true)]
    [string]$RedisLicenseFile,
    [Parameter(Mandatory = $true)]
    [switch]$RedisRedistributionApproved
)

$ErrorActionPreference = "Stop"
$LlamaTag = "b9637"
$LlamaCommit = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
$LlamaArchiveSha256 = "f7783c2b8c007f95e710ac40f26a24861a80b603b0b739fc54d7c926a4716c1e"
$LlamaArchiveUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/llama-$LlamaTag-bin-win-cpu-x64.zip"
$LlamaRepository = "https://github.com/ggml-org/llama.cpp.git"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VendorRoot = (Resolve-Path (Join-Path $PSScriptRoot "vendor")).Path
$ExpectedVendorRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "build\vendor"))

if ($VendorRoot -ne $ExpectedVendorRoot) {
    throw "Refusing to populate an unexpected vendor directory: $VendorRoot"
}
if (-not $RedisRedistributionApproved) {
    throw "Redis-compatible runtime redistribution approval is required."
}
$ResolvedRedis = (Resolve-Path -LiteralPath $RedisBinary).Path
$ResolvedRedisRuntime = (Resolve-Path -LiteralPath $RedisRuntimeDirectory).Path
$ResolvedRedisLicense = (Resolve-Path -LiteralPath $RedisLicenseFile).Path
if (-not [IO.Path]::IsPathFullyQualified($ResolvedRedis)) {
    throw "Redis binary must resolve to an absolute path."
}
if (-not (Get-Item -LiteralPath $ResolvedRedisRuntime).PSIsContainer) {
    throw "Redis runtime directory must be a directory."
}
if (-not $ResolvedRedis.StartsWith($ResolvedRedisRuntime + [IO.Path]::DirectorySeparatorChar)) {
    throw "Redis binary must be inside RedisRuntimeDirectory."
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("haradibots-vendor-" + [guid]::NewGuid())
$Stage = Join-Path $TemporaryRoot "stage"
$Archive = Join-Path $TemporaryRoot "llama.zip"
$Source = Join-Path $TemporaryRoot "llama-source"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

try {
    New-Item -ItemType Directory -Path $Stage -Force | Out-Null
    Invoke-WebRequest -Uri $LlamaArchiveUrl -OutFile $Archive
    if ((Get-Sha256 $Archive) -ne $LlamaArchiveSha256) {
        throw "llama.cpp release archive checksum mismatch."
    }
    Expand-Archive -LiteralPath $Archive -DestinationPath $Stage

    git clone --quiet --filter=blob:none --no-checkout $LlamaRepository $Source
    if ($LASTEXITCODE -ne 0) { throw "Unable to clone pinned llama.cpp source." }
    git -C $Source sparse-checkout init --no-cone
    git -C $Source sparse-checkout set convert_hf_to_gguf.py conversion gguf-py LICENSE
    git -C $Source checkout --quiet $LlamaCommit
    if ($LASTEXITCODE -ne 0) { throw "Unable to check out pinned llama.cpp commit." }
    $ResolvedCommit = (git -C $Source rev-parse HEAD).Trim()
    if ($ResolvedCommit -ne $LlamaCommit) {
        throw "llama.cpp source commit mismatch."
    }

    Copy-Item -LiteralPath (Join-Path $Source "convert_hf_to_gguf.py") -Destination $Stage
    Copy-Item -LiteralPath (Join-Path $Source "conversion") -Destination $Stage -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "gguf-py") -Destination $Stage -Recurse
    Copy-Item -LiteralPath (Join-Path $Source "LICENSE") -Destination (Join-Path $Stage "LICENSE.llama.cpp.txt")
    $RedisStage = Join-Path $Stage "redis"
    New-Item -ItemType Directory -Path $RedisStage | Out-Null
    Copy-Item -Path (Join-Path $ResolvedRedisRuntime "*") -Destination $RedisStage -Recurse
    Copy-Item -LiteralPath $ResolvedRedis -Destination (Join-Path $RedisStage "redis-server.exe") -Force
    Copy-Item -LiteralPath $ResolvedRedisLicense -Destination (Join-Path $Stage "LICENSE.redis-runtime.txt")

    $Assets = [ordered]@{}
    foreach ($Name in @("llama-cli", "llama-quantize", "llama-perplexity")) {
        $Path = Join-Path $Stage "$Name.exe"
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
            throw "llama.cpp archive is missing $Name.exe."
        }
        $Assets[$Name] = [ordered]@{
            sha256 = Get-Sha256 $Path
            source = $LlamaArchiveUrl
            license = "MIT"
        }
    }
    $Converter = Join-Path $Stage "convert_hf_to_gguf.py"
    $Assets["convert_hf_to_gguf.py"] = [ordered]@{
        sha256 = Get-Sha256 $Converter
        source = "https://github.com/ggml-org/llama.cpp/blob/$LlamaCommit/convert_hf_to_gguf.py"
        license = "MIT"
    }
    $RedisTarget = Join-Path $RedisStage "redis-server.exe"
    $Assets["redis-server"] = [ordered]@{
        sha256 = Get-Sha256 $RedisTarget
        source = $RedisSource
        license = $RedisLicense
    }
    $Files = [ordered]@{}
    Get-ChildItem -LiteralPath $Stage -Recurse -File | ForEach-Object {
        $Relative = $_.FullName.Substring($Stage.Length).TrimStart([char[]]"\/").Replace("\", "/")
        $Files[$Relative] = Get-Sha256 $_.FullName
    }
    $ManifestJson = [ordered]@{
        schema_version = "1"
        generated_utc = [DateTime]::UtcNow.ToString("o")
        assets = $Assets
        files = $Files
    } | ConvertTo-Json -Depth 5
    [IO.File]::WriteAllText(
        (Join-Path $Stage "vendor-manifest.json"),
        $ManifestJson,
        [Text.UTF8Encoding]::new($false)
    )

    Get-ChildItem -LiteralPath $VendorRoot -Force |
        Where-Object { $_.Name -ne "README.md" } |
        Remove-Item -Recurse -Force
    Copy-Item -Path (Join-Path $Stage "*") -Destination $VendorRoot -Recurse -Force
    & $Python -m build.verify_bundle
    if ($LASTEXITCODE -ne 0) { throw "Populated vendor verification failed." }
} finally {
    if (Test-Path -LiteralPath $TemporaryRoot) {
        $ResolvedTemporary = [IO.Path]::GetFullPath($TemporaryRoot)
        if ($ResolvedTemporary.StartsWith([IO.Path]::GetFullPath([IO.Path]::GetTempPath()))) {
            Remove-Item -LiteralPath $ResolvedTemporary -Recurse -Force
        }
    }
}
