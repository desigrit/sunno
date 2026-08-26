# Stages a self-contained Python backend for packaging.
#
# The venv's python.exe is a 268 KB shim that points at the python.org install via
# pyvenv.cfg, so it cannot simply be copied. This stages the real interpreter and overlays
# the venv's site-packages, omitting the CUDA payload entirely and other dead weight.
#
# The CUDA DLLs used to be staged here from packaging/cuda_allowlist.txt. They are now
# downloaded on demand instead; see the comment further down and server/cuda_download.py.

[CmdletBinding()]
param(
  [string]$Destination = "$PSScriptRoot\staging\backend",
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

# Interpreter core. Excludes docs, tcl/tk, the test suite and the base install's own
# site-packages — the venv's packages are overlaid below, and the base copy would otherwise
# drag pip/setuptools in behind them.
$excludeDirs = @("Doc", "tcl", "Lib\test", "Lib\tkinter", "Lib\idlelib", "Lib\lib2to3",
                 "Scripts", "Lib\site-packages")
robocopy $baseHome $py /E /NFL /NDL /NJH /NJS /NC /NS /NP `
  /XD $($excludeDirs | ForEach-Object { Join-Path $baseHome $_ }) `
  /XF "python*._pth" | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed staging the interpreter ($LASTEXITCODE)" }

# Overlay the venv's packages into the staged interpreter's site-packages.
$targetSite = Join-Path $py "Lib\site-packages"
New-Item -ItemType Directory -Path $targetSite -Force | Out-Null

Write-Host "Staging site-packages"
# Development-only packages that must not reach the package:
#   pip/setuptools/wheel - nothing installs at runtime
#   scipy (+ scipy.libs)  - 129 MB, replaced by the hand-rolled biquad in preprocess.py
#   PIL                   - icon generation only (packaging/make_icon.py)
#   pefile                - CUDA import analysis only (packaging/cuda_decide.py)
#   nvidia                - downloaded on demand, never packaged
$pkgExclude = @(
  "pip", "pip-*", "setuptools", "setuptools-*", "pkg_resources", "wheel", "wheel-*",
  "scipy", "scipy.libs", "scipy-*",
  "PIL", "pillow-*", "Pillow-*",
  "pefile", "pefile-*",
  "nvidia", "nvidia_*",
  "__pycache__"
)
# robocopy /XD takes a flat, space-separated list; passing a nested array silently drops
# entries, which is how scipy/PIL/pip leaked into the first package build.
$xd = @()
foreach ($name in $pkgExclude) {
  Get-ChildItem $site -Directory -Filter $name -ErrorAction SilentlyContinue |
    ForEach-Object { $xd += $_.FullName }
}
$xd += (Join-Path $site "nvidia")

$args = @($site, $targetSite, "/E", "/NFL", "/NDL", "/NJH", "/NJS", "/NC", "/NS", "/NP",
          "/XD") + $xd + @("__pycache__")
& robocopy @args | Out-Null
if ($LASTEXITCODE -ge 8) { throw "robocopy failed staging site-packages ($LASTEXITCODE)" }

# Fail loudly if anything dev-only still made it through, rather than shipping it quietly.
foreach ($banned in @("scipy", "PIL", "pip", "pefile")) {
  $leak = Join-Path $targetSite $banned
  if (Test-Path $leak) { throw "Dev-only package '$banned' leaked into the staged payload at $leak" }
}

# CUDA is no longer packaged. It was 828 MB, 61% of the payload, carried by every machine
# including the AMD and Intel ones that can never load it. It is now published as separate
# .xz assets on a pinned GitHub release and fetched on demand by server/cuda_download.py, on
# machines that both have a usable NVIDIA GPU and pick a model that needs one.
#
# What still has to ship is the manifest describing those files, because it is what the
# runtime checks a downloaded payload against. It lives in server/ rather than packaging/
# for the simple reason that only server/ and ui/ are staged below.
$manifest = Join-Path $repo "server\cuda_manifest.json"
if (-not (Test-Path $manifest)) {
  throw "Missing $manifest - run packaging/make_cuda_manifest.py before packaging."
}

# Backend source and the speaker model. Explicit include-list, not a directory copy, so a
# stray 100 MB benchmark model can't silently inflate the package.
Write-Host "Staging backend source"
foreach ($dir in @("server", "ui")) {
  $dst = Join-Path $Destination $dir
  # Copy-Item -Recurse nests the source inside an existing destination on a second run
  # (server\server), so clear it first.
  if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
  Copy-Item (Join-Path $repo $dir) $dst -Recurse -Force
}
Get-ChildItem (Join-Path $Destination "server") -Recurse -Filter "__pycache__" -Directory |
  Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

$models = Join-Path $Destination "models"
New-Item -ItemType Directory -Path $models -Force | Out-Null
# Explicit single file, not a directory copy: models/ also holds ~130 MB of benchmark
# models that must not silently inflate the package.
$speakerModel = "speaker-embedding-campplus-en.onnx"   # matches config.py's default
Copy-Item (Join-Path $repo "models\$speakerModel") (Join-Path $models $speakerModel) -Force

$size = (Get-ChildItem $Destination -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("`nStaged backend: {0:N0} MB at {1}" -f $size, $Destination)
