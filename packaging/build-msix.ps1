# Builds a signed MSIX for Sunno.
#
# Deliberately stops short of INSTALLING the package. Installing a self-signed MSIX requires
# adding the certificate to LocalMachine Trusted People, which is a machine-wide trust change
# and elevation prompt — not something to do unattended on someone's behalf. The script
# prints the two commands to run.
#
#   .\build-msix.ps1                 # full build
#   .\build-msix.ps1 -SkipStage      # reuse an existing staged backend

[CmdletBinding()]
param(
  [switch]$SkipStage,
  [switch]$SkipPublish,
  [string]$CertSubject = "CN=LiveCaptionsDev"
)

$ErrorActionPreference = "Stop"
$root     = Split-Path -Parent $PSScriptRoot
$staging  = Join-Path $PSScriptRoot "staging\package"
$out      = Join-Path $PSScriptRoot "out"
$appProj  = Join-Path $root "app\Sunno.csproj"

function Find-SdkTool([string]$name) {
  $pkg = Get-ChildItem "$env:USERPROFILE\.nuget\packages\microsoft.windows.sdk.buildtools" `
         -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
  if (-not $pkg) { throw "Microsoft.Windows.SDK.BuildTools not restored. Build app/ first." }
  $tool = Get-ChildItem $pkg.FullName -Recurse -Filter $name -ErrorAction SilentlyContinue |
          Where-Object { $_.FullName -match '\\x64\\' } | Select-Object -First 1
  if (-not $tool) { throw "$name not found under $($pkg.FullName)" }
  return $tool.FullName
}

$makeappx = Find-SdkTool "makeappx.exe"
$signtool = Find-SdkTool "signtool.exe"

New-Item -ItemType Directory -Path $out -Force | Out-Null

# Wipe the staging tree, but keep the staged backend when reusing it — it is the expensive
# part (~1.2 GB) and re-staging it dominates the build.
if (Test-Path $staging) {
  if ($SkipStage) {
    Get-ChildItem $staging -Force |
      Where-Object { $_.Name -ne "backend" } |
      Remove-Item -Recurse -Force
  } else {
    Remove-Item $staging -Recurse -Force
  }
}
New-Item -ItemType Directory -Path $staging -Force | Out-Null

# The staging directory is wiped above, so skipping the publish would produce a package with
# no application binaries — a footgun worth refusing outright.
if ($SkipPublish) {
  throw "-SkipPublish cannot be used: staging is rebuilt from scratch, so the app binaries " +
        "would be missing from the package. Use -SkipStage to reuse the staged backend instead."
}

# ---------------------------------------------------------------- app binaries
if (-not $SkipPublish) {
  Write-Host "Publishing the WinUI app..." -ForegroundColor Cyan
  # Published as a self-contained Win32 app and packaged externally with MakeAppx. Package
  # identity (which is what enables the per-app microphone toggle and AppCapability) comes
  # from being installed as an MSIX with EntryPoint="Windows.FullTrustApplication", not
  # from a build flag.
  dotnet publish $appProj -c Release -r win-x64 `
    -p:PackageForMsix=true `
    -p:PublishReadyToRun=false `
    -o "$staging" 2>&1 | Where-Object { $_ -match 'error|Error' }
  if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed ($LASTEXITCODE)" }

  # The Windows App SDK ships its own onnxruntime (Windows ML) at the package root. We never
  # call any ML API from C#, but a packaged process searches the package root for DLLs, so
  # sherpa-onnx's _sherpa_onnx.pyd bound to this copy instead of the 1.27 it ships beside
  # itself and died with an access violation the moment speaker labelling initialised. The
  # staged tree never hit it because only an installed package searches that directory.
  foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
    $victim = Join-Path $staging $dll
    if (Test-Path $victim) {
      Remove-Item $victim -Force
      Write-Host "  removed $dll (conflicts with sherpa-onnx)" -ForegroundColor DarkGray
    }
  }
}

# ---------------------------------------------------------------- backend
if (-not $SkipStage) {
  Write-Host "Staging the Python backend..." -ForegroundColor Cyan
  & (Join-Path $PSScriptRoot "stage-backend.ps1") `
      -Destination (Join-Path $staging "backend") -Clean
} else {
  $existing = Join-Path $staging "backend"
  if (-not (Test-Path $existing)) { throw "-SkipStage given but $existing does not exist." }

  # A cached backend is only safe to reuse if it still matches the source. Without this check a
  # cache that predates real bug fixes ships silently, because the package looks complete
  # either way.
  $stale = Get-ChildItem (Join-Path $root "server") -Filter *.py | Where-Object {
    $staged = Join-Path $existing "server\$($_.Name)"
    (-not (Test-Path $staged)) -or
    (Get-FileHash $_.FullName -Algorithm SHA256).Hash -ne (Get-FileHash $staged -Algorithm SHA256).Hash
  }
  if ($stale) {
    throw "-SkipStage refused: the cached backend is stale ($($stale.Name -join ', ')). " +
          "Re-run without -SkipStage."
  }

  Write-Host "Reusing staged backend (verified against server\*.py)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------- manifest + assets
Copy-Item (Join-Path $root "app\Package.appxmanifest") (Join-Path $staging "AppxManifest.xml") -Force

$assetsSrc = Join-Path $root "app\Assets"
$assetsDst = Join-Path $staging "Assets"
New-Item -ItemType Directory -Path $assetsDst -Force | Out-Null
Copy-Item "$assetsSrc\*" $assetsDst -Force

# The manifest names unqualified logo files; MakeAppx does not resolve scale variants for
# a loose (non-PRI) layout, so ensure the base names exist.
foreach ($pair in @(
    @("Square150x150Logo.scale-100.png", "Square150x150Logo.png"),
    @("Square44x44Logo.scale-100.png",   "Square44x44Logo.png"),
    @("Square71x71Logo.scale-100.png",   "Square71x71Logo.png"),
    @("Square310x310Logo.scale-100.png", "Square310x310Logo.png"),
    @("Wide310x150Logo.scale-100.png",   "Wide310x150Logo.png"),
    @("StoreLogo.scale-100.png",         "StoreLogo.png"),
    @("SplashScreen.scale-100.png",      "SplashScreen.png"))) {
  $src = Join-Path $assetsDst $pair[0]
  $dst = Join-Path $assetsDst $pair[1]
  if ((Test-Path $src) -and -not (Test-Path $dst)) { Copy-Item $src $dst -Force }
}

$size = (Get-ChildItem $staging -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
Write-Host ("Payload staged: {0:N0} MB" -f $size)

# ---------------------------------------------------------------- pack
$msix = Join-Path $out "Sunno.msix"
if (Test-Path $msix) { Remove-Item $msix -Force }

# A second onnxruntime.dll at the package root hijacks sherpa-onnx's own copy and crashes the
# backend with an access violation only once installed — never in the staged tree. Fail the
# build rather than ship a package whose speech engine dies on launch.
$rootOnnx = Get-ChildItem $staging -Filter "onnxruntime*.dll" -File -ErrorAction SilentlyContinue
if ($rootOnnx) {
  throw "onnxruntime DLLs at the package root would collide with sherpa-onnx: " +
        ($rootOnnx.Name -join ', ')
}

Write-Host "Packing..." -ForegroundColor Cyan
# MakeAppx emits a line per file (9k+). Redirect to a log rather than filtering the live
# pipeline: truncating that pipeline detaches the process mid-pack and leaves a 0-byte file.
$packLog = Join-Path $out "makeappx.log"
# Full manifest validation is on (no /nv): it verifies that every file the manifest
# references actually exists in the layout, which is exactly the class of mistake that
# would otherwise only surface at install time.
& $makeappx pack /d $staging /p $msix /o *> $packLog
if ($LASTEXITCODE -ne 0) {
  Get-Content $packLog -Tail 20 | ForEach-Object { Write-Host "  $_" }
  throw "makeappx failed ($LASTEXITCODE) - see $packLog"
}

# ---------------------------------------------------------------- sign
$cert = Get-ChildItem Cert:\CurrentUser\My |
        Where-Object { $_.Subject -eq $CertSubject } | Select-Object -First 1
if (-not $cert) {
  Write-Host "Creating self-signed certificate $CertSubject" -ForegroundColor Cyan
  $cert = New-SelfSignedCertificate -Type Custom -Subject $CertSubject `
            -KeyUsage DigitalSignature -FriendlyName "Sunno (development)" `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")
}

$pfx = Join-Path $out "LiveCaptionsDev.pfx"
$cer = Join-Path $out "LiveCaptionsDev.cer"
$pwd = ConvertTo-SecureString -String "Sunno-dev" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath $pfx -Password $pwd | Out-Null
Export-Certificate  -Cert $cert -FilePath $cer | Out-Null

& $signtool sign /fd SHA256 /a /f $pfx /p "Sunno-dev" $msix
if ($LASTEXITCODE -ne 0) { throw "signtool failed ($LASTEXITCODE)" }

$mb = (Get-Item $msix).Length / 1MB
Write-Host ""
Write-Host ("Signed package: {0}  ({1:N0} MB)" -f $msix, $mb) -ForegroundColor Green
Write-Host ""
Write-Host "To install (both steps need an elevated prompt):" -ForegroundColor Yellow
Write-Host "  1. Trust the development certificate — a machine-wide change:"
Write-Host "       Import-Certificate -FilePath `"$cer`" -CertStoreLocation Cert:\LocalMachine\TrustedPeople"
Write-Host "  2. Install the package:"
Write-Host "       Add-AppxPackage -Path `"$msix`""
Write-Host ""
Write-Host "  Uninstall with: Get-AppxPackage *Sunno* | Remove-AppxPackage"
