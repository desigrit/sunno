# Stages a self-contained Python backend for packaging.
#
# The venv's python.exe is a 268 KB shim that points at the python.org install via
# pyvenv.cfg, so it cannot simply be copied. This stages the real interpreter and overlays
# the venv's site-packages, omitting the CUDA DLLs that nothing imports (see
# packaging/cuda_allowlist.txt) and other dead weight.

[CmdletBinding()]
param(
  [string]$Destination = "$PSScriptRoot\staging\backend",
  [ValidateSet("x64", "arm64")]
  [string]$Architecture = "x64",
  [switch]$SkipCuda,
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$validationPython = $null

function Assert-FileHash([string]$Path, [string]$Expected) {
  $actual = (Get-FileHash $Path -Algorithm SHA256).Hash
  if ($actual -ne $Expected) {
    Remove-Item $Path -Force
    throw "Checksum mismatch for $Path (expected $Expected, got $actual)"
  }
}

if ($Clean -and (Test-Path $Destination)) { Remove-Item $Destination -Recurse -Force }
$py = Join-Path $Destination "python"
if (Test-Path $py) { Remove-Item $py -Recurse -Force }
New-Item -ItemType Directory -Path $py -Force | Out-Null

$targetSite = Join-Path $py "Lib\site-packages"
New-Item -ItemType Directory -Path $targetSite -Force | Out-Null

if ($Architecture -eq "arm64") {
  $pythonVersion = "3.12.10"
  $archiveName = "python-$pythonVersion-embed-arm64.zip"
  $cache = Join-Path $PSScriptRoot "cache"
  $archive = Join-Path $cache $archiveName
  $url = "https://www.python.org/ftp/python/$pythonVersion/$archiveName"
  New-Item -ItemType Directory -Path $cache -Force | Out-Null
  if (-not (Test-Path $archive)) {
    Write-Host "Downloading ARM64 embeddable Python $pythonVersion"
    Invoke-WebRequest -Uri $url -OutFile $archive
  }
  Assert-FileHash $archive "3065EFC3D382D1CDA66757AC71ADE11904FA6E350F5A97EB74811ACD71BA5532"

  Write-Host "Staging ARM64 embeddable Python $pythonVersion"
  Expand-Archive -Path $archive -DestinationPath $py -Force
  # Python 3.12.10's official ARM64 embeddable archive includes an unused x64 copy of this
  # runtime. No staged ARM64 binary imports it, and leaving it would make the payload mixed.
  Remove-Item (Join-Path $py "vcruntime140_1.dll") -Force
  # Embeddable Python disables site by default. Keep its isolated search path, but explicitly
  # admit the directory populated by the cross-platform pip install below.
  @("python312.zip", ".", "Lib\site-packages", "import site") |
    Set-Content (Join-Path $py "python312._pth") -Encoding ASCII

  $requirements = Join-Path $repo "requirements-arm64.txt"
  $hostPython = (Get-Command python -ErrorAction Stop).Source
  $hostVersion = & $hostPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if ($hostVersion -ne "3.12") {
    throw "ARM staging needs a Python 3.12 host for pip, found $hostVersion at $hostPython"
  }
  $validationPython = $hostPython
  Write-Host "Resolving native win_arm64 site-packages from requirements-arm64.txt"
  & $hostPython -m pip install --disable-pip-version-check --no-compile `
    --platform win_arm64 --implementation cp --python-version 3.12 --abi cp312 `
    --only-binary=:all: --require-hashes --target $targetSite -r $requirements
  if ($LASTEXITCODE -ne 0) { throw "pip failed staging ARM64 dependencies ($LASTEXITCODE)" }
  # Cross-platform pip launchers are generated for the build host and are never used at
  # runtime. Keeping them would introduce x64 executables into an otherwise native payload.
  Remove-Item (Join-Path $targetSite "bin") -Recurse -Force -ErrorAction SilentlyContinue

  # ONNX Runtime imports the ARM64 C++ runtime, which the embeddable Python archive does not
  # contain. Stage Microsoft's redistributable desktop VCLibs beside python.exe so a clean
  # Snapdragon machine does not depend on a machine-wide VC runtime installation.
  $vclibsName = "Microsoft.VCLibs.arm64.14.00.Desktop.zip"
  $vclibsArchive = Join-Path $cache $vclibsName
  if (-not (Test-Path $vclibsArchive)) {
    Write-Host "Downloading ARM64 Visual C++ runtime"
    Invoke-WebRequest -Uri "https://aka.ms/Microsoft.VCLibs.arm64.14.00.Desktop.appx" `
      -OutFile $vclibsArchive
  }
  Assert-FileHash $vclibsArchive "9A7F6D69EA6CF042EA8680B7CD0BFAA9C04F0F6CC89055D43F7F6CD0250508D3"
  $vclibs = Join-Path $cache "vclibs-arm64"
  if (Test-Path $vclibs) { Remove-Item $vclibs -Recurse -Force }
  Expand-Archive -Path $vclibsArchive -DestinationPath $vclibs
  foreach ($dll in @(
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll"
  )) {
    Copy-Item (Join-Path $vclibs $dll) (Join-Path $py $dll) -Force
  }
} else {
  $venv = Join-Path $repo ".venv"
  $site = Join-Path $venv "Lib\site-packages"

  # Resolve the real interpreter the venv was created from.
  $cfg = Get-Content (Join-Path $venv "pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' }
  $baseHome = ($cfg -split '=', 2)[1].Trim()
  if (-not (Test-Path (Join-Path $baseHome "python.exe"))) {
    throw "Base Python not found at '$baseHome' (from pyvenv.cfg)."
  }
  $validationPython = Join-Path $baseHome "python.exe"

  Write-Host "Staging interpreter from $baseHome"
  $excludeDirs = @("Doc", "tcl", "Lib\test", "Lib\tkinter", "Lib\idlelib", "Lib\lib2to3",
                   "Scripts", "Lib\site-packages")
  robocopy $baseHome $py /E /NFL /NDL /NJH /NJS /NC /NS /NP `
    /XD $($excludeDirs | ForEach-Object { Join-Path $baseHome $_ }) `
    /XF "python*._pth" | Out-Null
  if ($LASTEXITCODE -ge 8) { throw "robocopy failed staging the interpreter ($LASTEXITCODE)" }

  Write-Host "Staging site-packages"
  $pkgExclude = @(
    "pip", "pip-*", "setuptools", "setuptools-*", "pkg_resources", "wheel", "wheel-*",
    "scipy", "scipy.libs", "scipy-*",
    "PIL", "pillow-*", "Pillow-*",
    "pefile", "pefile-*",
    "nvidia", "nvidia_*",
    "__pycache__"
  )
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

  foreach ($banned in @("scipy", "PIL", "pip", "pefile")) {
    $leak = Join-Path $targetSite $banned
    if (Test-Path $leak) { throw "Dev-only package '$banned' leaked into the staged payload at $leak" }
  }

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
if (Test-Path $models) { Remove-Item $models -Recurse -Force }
New-Item -ItemType Directory -Path $models -Force | Out-Null
# Explicit single file, not a directory copy: models/ also holds ~130 MB of benchmark
# models that must not silently inflate the package.
$speakerModel = "speaker-embedding-campplus-en.onnx"   # matches config.py's default
if ($Architecture -eq "x64") {
  $speakerSource = Join-Path $repo "models\$speakerModel"
  if (-not (Test-Path $speakerSource)) {
    New-Item -ItemType Directory -Path (Split-Path $speakerSource) -Force | Out-Null
    Write-Host "Downloading the speaker embedding model"
    Invoke-WebRequest `
      -Uri "https://huggingface.co/openspeech/wespeaker-models/resolve/main/voxceleb_CAM%2B%2B_LM.onnx" `
      -OutFile $speakerSource
  }
  Assert-FileHash $speakerSource "1068E4AC3A76BB9C769E6816EF30BF89363F6E966F1D938210CB8ED4038F8E93"
  Copy-Item $speakerSource (Join-Path $models $speakerModel) -Force
}

Write-Host "Validating staged PE architectures"
& $validationPython (Join-Path $PSScriptRoot "verify_pe_arch.py") `
  $Destination --expected $Architecture
if ($LASTEXITCODE -ne 0) {
  throw "Staged backend contains binaries incompatible with $Architecture"
}

$size = (Get-ChildItem $Destination -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("`nStaged backend: {0:N0} MB at {1}" -f $size, $Destination)
