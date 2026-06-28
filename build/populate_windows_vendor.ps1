param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
$LlamaTag = "b9637"
$LlamaCommit = "aedb2a5e9ca3d4064148bbb919e0ddc0c1b70ab3"
$LlamaArchiveSha256 = "f7783c2b8c007f95e710ac40f26a24861a80b603b0b739fc54d7c926a4716c1e"
$LlamaArchiveUrl = "https://github.com/ggml-org/llama.cpp/releases/download/$LlamaTag/llama-$LlamaTag-bin-win-cpu-x64.zip"
$LlamaRepository = "https://github.com/ggml-org/llama.cpp.git"

$GarnetVersion = "1.1.10"
$GarnetCommit = "3986e6e654c693e87786e56f9ce1e61c4be06756"
$GarnetRepository = "https://github.com/microsoft/garnet.git"
$DotnetSdkVersion = "10.0.203"
$DotnetSdkUrl = "https://builds.dotnet.microsoft.com/dotnet/Sdk/$DotnetSdkVersion/dotnet-sdk-$DotnetSdkVersion-win-x64.zip"
$DotnetSdkSha512 = "41486bb422746154171e0ede7c1d0021605430728e71b83e94368184e8c97802f9cb5151be2cd1285936b11e452ddc0728ee3766db187d428e50531ac0c2161c"
$DotnetLibraryLicenseUrl = "https://dotnet.microsoft.com/dotnet_library_license.htm"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$VendorRoot = (Resolve-Path (Join-Path $PSScriptRoot "vendor")).Path
$CacheRoot = Join-Path $RepoRoot ".build-cache"
$CachedDotnetArchive = Join-Path $CacheRoot "dotnet-sdk-$DotnetSdkVersion-win-x64.zip"
$ExpectedVendorRoot = [IO.Path]::GetFullPath((Join-Path $RepoRoot "build\vendor"))
if ($VendorRoot -ne $ExpectedVendorRoot) {
    throw "Refusing to populate an unexpected vendor directory: $VendorRoot"
}

$TemporaryRoot = Join-Path ([IO.Path]::GetTempPath()) ("haradibots-vendor-" + [guid]::NewGuid())
$Stage = Join-Path $TemporaryRoot "stage"
$LlamaArchive = Join-Path $TemporaryRoot "llama.zip"
$LlamaSource = Join-Path $TemporaryRoot "llama-source"
$DotnetArchive = Join-Path $TemporaryRoot "dotnet-sdk.zip"
$DotnetRoot = Join-Path $TemporaryRoot "dotnet"
$GarnetSource = Join-Path $TemporaryRoot "garnet-source"
$GarnetPublish = Join-Path $TemporaryRoot "garnet-publish"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Assert-GitSuccess([string]$Message) {
    if ($LASTEXITCODE -ne 0) { throw $Message }
}

try {
    New-Item -ItemType Directory -Path $Stage, $DotnetRoot, $CacheRoot -Force | Out-Null

    Invoke-WebRequest -Uri $LlamaArchiveUrl -OutFile $LlamaArchive
    if ((Get-Sha256 $LlamaArchive) -ne $LlamaArchiveSha256) {
        throw "llama.cpp release archive checksum mismatch."
    }
    Expand-Archive -LiteralPath $LlamaArchive -DestinationPath $Stage
    git clone --quiet --filter=blob:none --no-checkout $LlamaRepository $LlamaSource
    Assert-GitSuccess "Unable to clone pinned llama.cpp source."
    git -C $LlamaSource sparse-checkout init --no-cone
    Assert-GitSuccess "Unable to initialize llama.cpp sparse checkout."
    git -C $LlamaSource sparse-checkout set convert_hf_to_gguf.py conversion gguf-py LICENSE
    Assert-GitSuccess "Unable to select llama.cpp converter sources."
    git -C $LlamaSource checkout --quiet $LlamaCommit
    Assert-GitSuccess "Unable to check out pinned llama.cpp commit."
    if ((git -C $LlamaSource rev-parse HEAD).Trim() -ne $LlamaCommit) {
        throw "llama.cpp source commit mismatch."
    }
    Copy-Item (Join-Path $LlamaSource "convert_hf_to_gguf.py") $Stage
    Copy-Item (Join-Path $LlamaSource "conversion") $Stage -Recurse
    Copy-Item (Join-Path $LlamaSource "gguf-py") $Stage -Recurse
    Copy-Item (Join-Path $LlamaSource "LICENSE") (Join-Path $Stage "LICENSE.llama.cpp.txt")

    if (-not (Test-Path $CachedDotnetArchive -PathType Leaf) -or
        (Get-FileHash $CachedDotnetArchive -Algorithm SHA512).Hash.ToLowerInvariant() -ne $DotnetSdkSha512) {
        Invoke-WebRequest -Uri $DotnetSdkUrl -OutFile $CachedDotnetArchive
    }
    $SdkHash = (Get-FileHash $CachedDotnetArchive -Algorithm SHA512).Hash.ToLowerInvariant()
    if ($SdkHash -ne $DotnetSdkSha512) {
        throw ".NET SDK archive checksum mismatch."
    }
    Copy-Item $CachedDotnetArchive $DotnetArchive
    Expand-Archive -LiteralPath $DotnetArchive -DestinationPath $DotnetRoot

    git clone --quiet --filter=blob:none $GarnetRepository $GarnetSource
    Assert-GitSuccess "Unable to clone Garnet source."
    git -C $GarnetSource checkout --quiet $GarnetCommit
    Assert-GitSuccess "Unable to check out pinned Garnet commit."
    if ((git -C $GarnetSource rev-parse HEAD).Trim() -ne $GarnetCommit) {
        throw "Garnet source commit mismatch."
    }
    $Dotnet = Join-Path $DotnetRoot "dotnet.exe"
    $env:DOTNET_CLI_TELEMETRY_OPTOUT = "1"
    $env:DOTNET_SKIP_FIRST_TIME_EXPERIENCE = "1"
    $env:DOTNET_NOLOGO = "1"
    & $Dotnet publish (Join-Path $GarnetSource "main\GarnetServer\GarnetServer.csproj") `
        --configuration Release `
        --framework net10.0 `
        --runtime win-x64 `
        --self-contained true `
        --output $GarnetPublish `
        -p:PublishReadyToRun=true
    if ($LASTEXITCODE -ne 0) { throw "Self-contained Garnet publish failed." }

    $GarnetStage = Join-Path $Stage "garnet"
    New-Item -ItemType Directory -Path $GarnetStage | Out-Null
    Copy-Item (Join-Path $GarnetPublish "*") $GarnetStage -Recurse
    Copy-Item (Join-Path $GarnetSource "LICENSE") (Join-Path $GarnetStage "LICENSE.Garnet-MIT.txt")
    Copy-Item (Join-Path $GarnetSource "NOTICE.md") (Join-Path $GarnetStage "NOTICE.Garnet.md")
    Invoke-WebRequest -Uri $DotnetLibraryLicenseUrl -OutFile (Join-Path $GarnetStage "LICENSE.dotnet-library.html")
    Copy-Item (Join-Path $DotnetRoot "ThirdPartyNotices.txt") (Join-Path $GarnetStage "NOTICE.dotnet-third-party.txt")

    $GarnetExecutable = Join-Path $GarnetStage "GarnetServer.exe"
    if (-not (Test-Path $GarnetExecutable -PathType Leaf)) {
        throw "Self-contained Garnet publish produced no executable."
    }

    $Assets = [ordered]@{}
    foreach ($Name in @("llama-cli", "llama-completion", "llama-quantize", "llama-perplexity")) {
        $Path = Join-Path $Stage "$Name.exe"
        if (-not (Test-Path $Path -PathType Leaf)) {
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
    $Assets["garnet-server"] = [ordered]@{
        sha256 = Get-Sha256 $GarnetExecutable
        source = "https://github.com/microsoft/garnet/tree/$GarnetCommit"
        license = "MIT; self-contained Windows .NET runtime under .NET Library License"
        version = $GarnetVersion
        dotnet_sdk = $DotnetSdkVersion
    }

    $Files = [ordered]@{}
    Get-ChildItem $Stage -Recurse -File | ForEach-Object {
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

    Get-ChildItem $VendorRoot -Force |
        Where-Object { $_.Name -ne "README.md" } |
        Remove-Item -Recurse -Force
    Copy-Item (Join-Path $Stage "*") $VendorRoot -Recurse -Force
    & $Python -m build.verify_bundle
    if ($LASTEXITCODE -ne 0) { throw "Populated vendor verification failed." }
} finally {
    if (Test-Path $TemporaryRoot) {
        $ResolvedTemporary = [IO.Path]::GetFullPath($TemporaryRoot)
        $ResolvedTempBase = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
        if ($ResolvedTemporary.StartsWith($ResolvedTempBase)) {
            Start-Sleep -Seconds 1
            Remove-Item $ResolvedTemporary -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}
