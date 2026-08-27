# Stages a self-contained Python backend for packaging.
#
# Two architectures, two genuinely different mechanisms, deliberately not threaded through
# one code path:
#
#   x64    The venv's python.exe is a 268 KB shim that points at the python.org install via
#          pyvenv.cfg, so it cannot simply be copied. This stages the real interpreter and
#          overlays the venv's site-packages. That interpreter is a *full install*, which has
#          no `._pth` and must not gain one, hence the /XF below.
#
#   arm64  There is no ARM64 venv to copy on an x64 build host, so the tree is assembled
#          instead: python.org's ARM64 *embeddable* archive, plus wheels resolved for
#          win_arm64 by pip. An embeddable tree is the mirror image of a full install - its
#          `python312._pth` is what defines sys.path, so it must be kept AND extended, or the
#          interpreter starts and cannot find Lib\site-packages. Dropping that file the way
#          the x64 path does would produce an ARM package that fails to import anything, on
#          the one machine the developer cannot test.
#
# The CUDA DLLs used to be staged here from packaging/cuda_allowlist.txt. They are now
# downloaded on demand instead; see the comment further down and server/cuda_download.py.

[CmdletBinding()]
param(
  [string]$Destination = "$PSScriptRoot\staging\backend",
  [ValidateSet("x64", "arm64")]
  [string]$Architecture = "x64",
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $repo ".venv"
$site = Join-Path $venv "Lib\site-packages"

# Python version for the ARM64 embeddable. Pinned rather than derived from the build host,
# because the wheels in requirements-arm64.txt are cp312 and a mismatch would install
# silently and fail at import.
$armPythonVersion = "3.12.10"

if ($Clean -and (Test-Path $Destination)) { Remove-Item $Destination -Recurse -Force }
$py = Join-Path $Destination "python"
New-Item -ItemType Directory -Path $py -Force | Out-Null

if ($Architecture -eq "arm64") {
  # ---------------------------------------------------------------- ARM64 --
  $cache = Join-Path $PSScriptRoot "cache"
  New-Item -ItemType Directory -Path $cache -Force | Out-Null
  $zipName = "python-$armPythonVersion-embed-arm64.zip"
  $zipPath = Join-Path $cache $zipName
  if (-not (Test-Path $zipPath)) {
    $url = "https://www.python.org/ftp/python/$armPythonVersion/$zipName"
    Write-Host "Downloading ARM64 embeddable Python $armPythonVersion"
    Invoke-WebRequest -Uri $url -OutFile $zipPath -UseBasicParsing
  }
  Write-Host "Staging ARM64 embeddable Python $armPythonVersion"
  Expand-Archive $zipPath -DestinationPath $py -Force

  # The embeddable archive ships an x64 vcruntime140_1.dll. Nothing ARM64 imports it -
  # onnxruntime's own loader skips that DLL when the machine is ARM64 - and leaving it makes
  # the payload mixed-architecture, which packaging/verify_pe_arch.py rejects.
  $strayVc = Join-Path $py "vcruntime140_1.dll"
  if (Test-Path $strayVc) { Remove-Item $strayVc -Force }

  # sys.path for an embeddable tree comes from this file and nowhere else. Without the
  # site-packages line the interpreter runs but imports nothing we install below.
  $pth = Get-ChildItem $py -Filter "python*._pth" | Select-Object -First 1
  if (-not $pth) { throw "No python*._pth in the embeddable archive - layout changed." }
  $pthLines = @(Get-Content $pth.FullName)
  if ($pthLines -notcontains "Lib\site-packages") {
    $insertAt = [Array]::IndexOf($pthLines, ".")
    if ($insertAt -lt 0) { $pthLines = @("Lib\site-packages") + $pthLines }
    else { $pthLines = $pthLines[0..$insertAt] + @("Lib\site-packages") + $pthLines[($insertAt + 1)..($pthLines.Count - 1)] }
    Set-Content $pth.FullName ($pthLines -join "`r`n") -Encoding ASCII
  }

  $targetSite = Join-Path $py "Lib\site-packages"
  New-Item -ItemType Directory -Path $targetSite -Force | Out-Null

  $requirements = Join-Path $repo "requirements-arm64.txt"
  if (-not (Test-Path $requirements)) { throw "Missing $requirements" }
  Write-Host "Resolving native win_arm64 wheels from requirements-arm64.txt"
  & (Join-Path $venv "Scripts\python.exe") -m pip install --no-cache-dir --only-binary=:all: `
      --require-hashes --platform win_arm64 --python-version 3.12 --implementation cp --abi cp312 `
      --target $targetSite -r $requirements | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "pip failed staging ARM64 dependencies ($LASTEXITCODE)" }

  # Cross-platform pip emits console launchers for the BUILD host, so a win_arm64 --target
  # install leaves x64 .exe files behind. They are build-time entry points that nothing at
  # runtime imports, and they are the difference between a clean ARM payload and a mixed one.
  $binDir = Join-Path $targetSite "bin"
  if (Test-Path $binDir) {
    $launchers = (Get-ChildItem $binDir -File).Count
    Remove-Item $binDir -Recurse -Force
    Write-Host "Removed $launchers build-host launchers from site-packages\bin"
  }

  # --target also drops a Scripts\ tree that a normal install would not: 22 MB of link-time
  # .lib files and a SECOND copy of onnxruntime.dll and the sherpa-onnx DLLs. Nothing loads
  # from there, but build-msix.ps1 already fails the build over duplicate onnxruntime copies
  # hijacking sherpa-onnx's own, and a third copy sitting inside the payload is exactly the
  # kind of thing that guard exists to prevent - it simply does not look here.
  $scriptsDir = Join-Path $targetSite "Scripts"
  if (Test-Path $scriptsDir) {
    $mb = (Get-ChildItem $scriptsDir -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
    Remove-Item $scriptsDir -Recurse -Force
    Write-Host ("Removed {0:N0} MB of --target Scripts artifacts (duplicate onnxruntime + .lib)" -f $mb)
  }

  # ONNX Runtime loads msvcp140.dll, which the embeddable archive does not carry (it ships
  # vcruntime140.dll only). It comes from the ARM64 VCLibs redistributable.
  $vclibsName = "Microsoft.VCLibs.arm64.14.00.Desktop.zip"
  $vclibsPath = Join-Path $cache $vclibsName
  if (-not (Test-Path $vclibsPath)) {
    Write-Host "Downloading $vclibsName"
    Invoke-WebRequest -Uri "https://aka.ms/Microsoft.VCLibs.arm64.14.00.Desktop.appx" `
      -OutFile $vclibsPath -UseBasicParsing
  }
  $vcExtract = Join-Path $cache "vclibs-arm64"
  if (Test-Path $vcExtract) { Remove-Item $vcExtract -Recurse -Force }
  Expand-Archive $vclibsPath -DestinationPath $vcExtract -Force
  $msvcp = Get-ChildItem $vcExtract -Recurse -Filter "msvcp140.dll" | Select-Object -First 1
  if (-not $msvcp) { throw "msvcp140.dll not found in $vclibsName - ONNX Runtime will not load." }
  Copy-Item $msvcp.FullName (Join-Path $py "msvcp140.dll") -Force
  Write-Host "Staged ARM64 msvcp140.dll for ONNX Runtime"
}
else {
  # ------------------------------------------------------------------ x64 --
  # Resolve the real interpreter the venv was created from.
  $cfg = Get-Content (Join-Path $venv "pyvenv.cfg") | Where-Object { $_ -match '^home\s*=' }
  $baseHome = ($cfg -split '=', 2)[1].Trim()
  if (-not (Test-Path (Join-Path $baseHome "python.exe"))) {
    throw "Base Python not found at '$baseHome' (from pyvenv.cfg)."
  }

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
