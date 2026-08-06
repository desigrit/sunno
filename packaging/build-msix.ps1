# Builds signed x64 and ARM64 MSIX packages plus a signed bundle containing both.
#
# Deliberately stops short of installing the package. Trusting a self-signed certificate is a
# machine-wide change and needs an elevated prompt, so the script prints the commands instead.
#
#   .\build-msix.ps1

[CmdletBinding()]
param(
  [switch]$SkipPublish,
  # Must equal Package/Identity/Publisher in app/Package.appxmanifest.
  [string]$CertSubject = "CN=A2015C41-8111-42CA-8A27-273B3309C099"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$stagingRoot = Join-Path $PSScriptRoot "staging"
$out = Join-Path $PSScriptRoot "out"
$appProj = Join-Path $root "app\Sunno.csproj"
$manifestSource = Join-Path $root "app\Package.appxmanifest"

function Find-SdkTool([string]$name) {
  $pkg = Get-ChildItem "$env:USERPROFILE\.nuget\packages\microsoft.windows.sdk.buildtools" `
         -Directory -ErrorAction SilentlyContinue | Sort-Object Name | Select-Object -Last 1
  if (-not $pkg) { throw "Microsoft.Windows.SDK.BuildTools not restored. Build app/ first." }
  # This is the architecture of the build host, not the package target.
  $tool = Get-ChildItem $pkg.FullName -Recurse -Filter $name -ErrorAction SilentlyContinue |
          Where-Object { $_.FullName -match '\\x64\\' } | Select-Object -First 1
  if (-not $tool) { throw "$name not found under $($pkg.FullName)" }
  return $tool.FullName
}

function Reset-PackageStaging([string]$staging) {
  if (Test-Path $staging) {
    Remove-Item $staging -Recurse -Force
  }
  New-Item -ItemType Directory -Path $staging -Force | Out-Null
}

function Copy-Assets([string]$staging) {
  $assetsSrc = Join-Path $root "app\Assets"
  $assetsDst = Join-Path $staging "Assets"
  New-Item -ItemType Directory -Path $assetsDst -Force | Out-Null
  Copy-Item "$assetsSrc\*" $assetsDst -Force

  # MakeAppx does not resolve scale variants for a loose layout.
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
}

if ($SkipPublish) {
  throw "-SkipPublish cannot be used: package staging is rebuilt from scratch, so application " +
        "binaries would be missing."
}

$makeappx = Find-SdkTool "makeappx.exe"
$signtool = Find-SdkTool "signtool.exe"
$validationPython = (Get-Command python -ErrorAction Stop).Source
$peVerifier = Join-Path $PSScriptRoot "verify_pe_arch.py"
$architectures = @(
  @{ Name = "x64"; Runtime = "win-x64"; Platform = "x64" },
  @{ Name = "arm64"; Runtime = "win-arm64"; Platform = "ARM64" }
)

New-Item -ItemType Directory -Path $out -Force | Out-Null
$bundleInput = Join-Path $stagingRoot "bundle-input"
if (Test-Path $bundleInput) { Remove-Item $bundleInput -Recurse -Force }
New-Item -ItemType Directory -Path $bundleInput -Force | Out-Null

$packages = @()
foreach ($target in $architectures) {
  $architecture = $target.Name
  $runtime = $target.Runtime
  $platform = $target.Platform
  $staging = Join-Path $stagingRoot "package-$architecture"
  Reset-PackageStaging $staging

  Write-Host "Publishing the $architecture WinUI app..." -ForegroundColor Cyan
  dotnet publish $appProj -c Release -r $runtime `
    -p:Platform=$platform `
    -p:PackageForMsix=true `
    -p:PublishReadyToRun=false `
    -o "$staging" 2>&1 | Where-Object { $_ -match 'error|Error' }
  if ($LASTEXITCODE -ne 0) { throw "dotnet publish failed for $architecture ($LASTEXITCODE)" }

  # Windows App SDK's root ONNX Runtime can hijack the backend's own version.
  foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
    $victim = Join-Path $staging $dll
    if (Test-Path $victim) {
      Remove-Item $victim -Force
      Write-Host "  removed $dll (conflicts with backend runtime)" -ForegroundColor DarkGray
    }
  }

  $backend = Join-Path $staging "backend"
  Write-Host "Staging the $architecture Python backend..." -ForegroundColor Cyan
  & (Join-Path $PSScriptRoot "stage-backend.ps1") `
    -Destination $backend -Architecture $architecture -Clean

  $manifest = [IO.File]::ReadAllText($manifestSource)
  $manifest = $manifest.Replace(
    'ProcessorArchitecture="x64"',
    "ProcessorArchitecture=`"$architecture`""
  )
  if ($manifest -notmatch "ProcessorArchitecture=`"$architecture`"") {
    throw "Could not template ProcessorArchitecture for $architecture."
  }
  [IO.File]::WriteAllText(
    (Join-Path $staging "AppxManifest.xml"),
    $manifest,
    [Text.UTF8Encoding]::new($false)
  )
  Copy-Assets $staging

  $rootOnnx = Get-ChildItem $staging -Filter "onnxruntime*.dll" -File `
                -ErrorAction SilentlyContinue
  if ($rootOnnx) {
    throw "ONNX Runtime DLLs at the package root would collide with the backend: " +
          ($rootOnnx.Name -join ', ')
  }

  # Windows App SDK deliberately includes this ARM helper in its win-x64 runtime pack so an x64
  # app can expose workload resources under ARM64EC. The same hash appears in Microsoft's x64 and
  # arm64ec NuGet assets; it is not a target-selection leak from our publish.
  $verifyArgs = @($peVerifier, $staging, "--expected", $architecture)
  if ($architecture -eq "x64") {
    $verifyArgs += @(
      "--allow-cross-arch-file",
      "Microsoft.Windows.Workloads.Resources_ec.dll",
      "arm64",
      "3DBEFA883EA9DCBB0FEE463AFDA0121E385BB678652CA2FAA19CF2ABF517091E"
    )
  }
  & $validationPython @verifyArgs
  if ($LASTEXITCODE -ne 0) {
    throw "The complete $architecture package contains incompatible binaries."
  }

  $size = (Get-ChildItem $staging -Recurse -File | Measure-Object Length -Sum).Sum / 1MB
  Write-Host ("$architecture payload staged: {0:N0} MB" -f $size)

  $msix = Join-Path $out "Sunno-$architecture.msix"
  if (Test-Path $msix) { Remove-Item $msix -Force }
  $packLog = Join-Path $out "makeappx-$architecture.log"
  & $makeappx pack /d $staging /p $msix /o *> $packLog
  if ($LASTEXITCODE -ne 0) {
    Get-Content $packLog -Tail 20 | ForEach-Object { Write-Host "  $_" }
    throw "makeappx failed for $architecture ($LASTEXITCODE) - see $packLog"
  }
  Copy-Item $msix (Join-Path $bundleInput (Split-Path $msix -Leaf)) -Force
  $packages += $msix
}

# MakeAppx cannot bundle signed packages. Build the bundle first, then sign both the bundle and
# the standalone architecture packages.
$bundle = Join-Path $out "Sunno.msixbundle"
if (Test-Path $bundle) { Remove-Item $bundle -Force }
$bundleLog = Join-Path $out "makeappx-bundle.log"
[xml]$sourceManifest = Get-Content $manifestSource
$bundleVersion = $sourceManifest.Package.Identity.Version
& $makeappx bundle /d $bundleInput /p $bundle /bv $bundleVersion /o *> $bundleLog
if ($LASTEXITCODE -ne 0) {
  Get-Content $bundleLog -Tail 20 | ForEach-Object { Write-Host "  $_" }
  throw "makeappx bundle failed ($LASTEXITCODE) - see $bundleLog"
}

$cert = Get-ChildItem Cert:\CurrentUser\My |
        Where-Object { $_.Subject -eq $CertSubject } | Select-Object -First 1
if (-not $cert) {
  Write-Host "Creating self-signed certificate $CertSubject" -ForegroundColor Cyan
  $cert = New-SelfSignedCertificate -Type Custom -Subject $CertSubject `
            -KeyUsage DigitalSignature -FriendlyName "Sunno (development)" `
            -CertStoreLocation "Cert:\CurrentUser\My" `
            -TextExtension @("2.5.29.37={text}1.3.6.1.5.5.7.3.3", "2.5.29.19={text}")
}

$pfx = Join-Path $out "SunnoDev.pfx"
$cer = Join-Path $out "SunnoDev.cer"
$pwd = ConvertTo-SecureString -String "Sunno-dev" -Force -AsPlainText
Export-PfxCertificate -Cert $cert -FilePath $pfx -Password $pwd | Out-Null
Export-Certificate -Cert $cert -FilePath $cer | Out-Null

foreach ($artifact in @($bundle) + $packages) {
  & $signtool sign /fd SHA256 /a /f $pfx /p "Sunno-dev" $artifact
  if ($LASTEXITCODE -ne 0) { throw "signtool failed for $artifact ($LASTEXITCODE)" }
}

$mb = (Get-Item $bundle).Length / 1MB
Write-Host ""
Write-Host ("Signed bundle: {0}  ({1:N0} MB)" -f $bundle, $mb) -ForegroundColor Green
Write-Host ""
Write-Host "To install (both steps need an elevated prompt):" -ForegroundColor Yellow
Write-Host "  1. Trust the development certificate:"
Write-Host "       Import-Certificate -FilePath `"$cer`" -CertStoreLocation Cert:\LocalMachine\TrustedPeople"
Write-Host "  2. Install the architecture-selecting bundle:"
Write-Host "       Add-AppxPackage -Path `"$bundle`""
Write-Host ""
Write-Host "  Uninstall with: Get-AppxPackage *Sunno* | Remove-AppxPackage"
