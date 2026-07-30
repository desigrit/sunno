# Stages a self-contained Python backend for packaging.
#
# The venv's python.exe is a 268 KB shim that points at the python.org install via
# pyvenv.cfg, so it cannot simply be copied. This stages the real interpreter and overlays
# the venv's site-packages, omitting the CUDA DLLs that nothing imports (see
# packaging/cuda_allowlist.txt) and other dead weight.

[CmdletBinding()]
param(
  [string]$Destination = "$PSScriptRoot\staging\backend",
  [switch]$SkipCuda,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo ".venv"
$site = Join-Path $venv "Lib\site-packages"

# Resolve the real interpreter the venv was created from.
$cfg = Get-Content (Join-Path $venv "pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' }
$baseHome = ($cfg -split '=', 2)[1].Trim()
if (-not (Test-Path (Join-Path $baseHome "python.exe"))) {
  throw "Base Python not found at '$baseHome' (from pyvenv.cfg)."
}

if ($Clean -and (Test-Path $Destination)) { Remove-Item $Destination -Recurse -Force }
$py = Join-Path $Destination "python"
New-Item -ItemType Directory -Path $py -Force | Out-Null

Write-Host "Staging interpreter from $baseHome"

# Interpreter core. Excludes docs, tcl/tk, the test suite and bundled installers - none are
# reachable from the backend and they are ~60 MB.
$excludeDirs = @("Doc", "tcl", "Lib\test", "Lib\tkinter", "Lib\idlelib", "Lib\lib2to3", "Scripts")
robocopy $baseHome $py /E /NFL /NDL /NJH /NJS /NC /NS /NP `
  /XD $($excludeDirs | ForEach-Object { Join-Path $baseHome $_ }) `
  /XF "python*._pth" | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed staging the interpreter ($LASTEXITCODE)" }

# Overlay the venv's packages into the staged interpreter's site-packages.
$targetSite = Join-Path $py "Lib\site-packages"
New-Item -ItemType Directory -Path $targetSite -Force | Out-Null

Write-Host "Staging site-packages"
$pkgExclude = @("pip", "setuptools", "pkg_resources", "wheel", "pefile", "__pycache__")
robocopy $site $targetSite /E /NFL /NDL /NJH /NJS /NC /NS /NP `
  /XD $($pkgExclude | ForEach-Object { Join-Path $site $_ }) nvidia | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed staging site-packages ($LASTEXITCODE)" }

# CUDA: copy only what the import graph proves is reachable.
if (-not $SkipCuda) {
  $allow = Join-Path $PSScriptRoot "cuda_allowlist.txt"
  if (-not (Test-Path $allow)) { throw "Missing $allow - run cuda_decide.py first." }
  $wanted = Get-Content $allow | Where-Object { $_.Trim() }
  Write-Host "Staging $($wanted.Count) CUDA DLLs from the allow-list"
  foreach ($rel in $wanted) {
    $src = Join-Path $site "nvidia\$($rel -replace '/', '\')"
    $dst = Join-Path $targetSite "nvidia\$($rel -replace '/', '\')"
    if (-not (Test-Path $src)) { throw "Allow-listed DLL missing: $src" }
    New-Item -ItemType Directory -Path (Split-Path $dst) -Force | Out-Null
    Copy-Item $src $dst -Force
  }
}

# Backend source and the speaker model. Explicit include-list, not a directory copy, so a
# stray 100 MB benchmark model can't silently inflate the package.
Write-Host "Staging backend source"
Copy-Item (Join-Path $repo "server") (Join-Path $Destination "server") -Recurse -Force
Get-ChildItem (Join-Path $Destination "server") -Recurse -Filter "__pycache__" -Directory |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

# The web UI is served over HTTP for the phone/handheld path; server/app.py serves it
# unconditionally, so omitting it would 404 that route.
Copy-Item (Join-Path $repo "ui") (Join-Path $Destination "ui") -Recurse -Force

$models = Join-Path $Destination "models"
New-Item -ItemType Directory -Path $models -Force | Out-Null
$speakerModel = "wespeaker_en_voxceleb_CAM++_LM.onnx"   # config.py default; the others are benchmarks
Copy-Item (Join-Path $repo "models\$speakerModel") (Join-Path $models $speakerModel) -Force

$size = (Get-ChildItem $Destination -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("`nStaged backend: {0:N0} MB at {1}" -f $size, $Destination)
