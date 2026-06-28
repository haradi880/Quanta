param(
    [string]$Python = "python",
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location $RepoRoot
try {
    & $Python -m build.verify_bundle
    if ($LASTEXITCODE -ne 0) { throw "Native vendor verification failed." }

    $arguments = @("-m", "PyInstaller", "--noconfirm")
    if ($Clean) { $arguments += "--clean" }
    $arguments += (Join-Path $RepoRoot "build\fat_binary.spec")
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

    & (Join-Path $RepoRoot "dist\HaradiBots\HaradiBots.exe") "--help"
    if ($LASTEXITCODE -ne 0) { throw "Packaged CLI smoke test failed." }
    & (Join-Path $RepoRoot "dist\HaradiBots\HaradiBots.exe") "doctor" "--json"
    if ($LASTEXITCODE -ne 0) { throw "Packaged native runtime check failed." }
} finally {
    Pop-Location
}
