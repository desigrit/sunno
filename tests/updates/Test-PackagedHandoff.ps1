# Run with Windows Developer Mode enabled. Register a disposable full-trust app
# so the write uses the same MSIX redirection rules as an installed Sunno.
$ErrorActionPreference = 'Stop'
$work = Join-Path ([IO.Path]::GetTempPath()) ('SunnoUpdate-' + [Guid]::NewGuid().ToString('N'))
if ($env:RUNNER_TEMP) { $work = Join-Path $env:RUNNER_TEMP (Split-Path $work -Leaf) }
$stage = Join-Path $work 'app'
$report = Join-Path $work 'handoff.json'
$app = Join-Path $PSScriptRoot '../../app'
$name = 'Sunno.UpdatePathCheck'
if (Get-AppxPackage -Name $name) { throw "The test package is already registered: $name" }

dotnet publish (Join-Path $PSScriptRoot 'UpdateTests.csproj') -c Release -r win-x64 --self-contained false -p:PackagedHandoff=true -o $stage
if ($LASTEXITCODE) { throw 'Could not build the packaged updater checks' }
Copy-Item -LiteralPath (Join-Path $app 'Assets') -Destination (Join-Path $stage 'Assets') -Recurse
Copy-Item -LiteralPath (Join-Path $stage 'Assets/SplashScreen.scale-100.png') -Destination (Join-Path $stage 'Assets/SplashScreen.png') -Force
[xml]$manifest = Get-Content -Raw -LiteralPath (Join-Path $app 'Package.appxmanifest')
$manifest.Package.Identity.SetAttribute('Name', $name)
$manifest.Package.Identity.SetAttribute('Publisher', "CN=$name")
$manifest.Package.Applications.Application.SetAttribute('Executable', 'UpdateTests.exe')
$manifestPath = Join-Path $stage 'AppxManifest.xml'
$manifest.Save($manifestPath)

Add-AppxPackage -Register $manifestPath
$package = Get-AppxPackage -Name $name -ErrorAction Stop
try {
    Invoke-CommandInDesktopPackage -PackageFamilyName $package.PackageFamilyName -AppId App -Command (Join-Path $stage 'UpdateTests.exe') -Args ('--packaged-handoff "' + $report + '"')
    $deadline = (Get-Date).AddSeconds(45)
    while (!(Test-Path -LiteralPath $report) -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 200 }
    if (!(Test-Path -LiteralPath $report)) { throw 'The process with package identity did not return an installer path' }

    # This process has no package identity, just like a separate installer.
    $result = Get-Content -Raw -LiteralPath $report | ConvertFrom-Json
    $hash = Get-FileHash -LiteralPath $result.installerPath -Algorithm SHA256
    if ($hash.Hash -ne $result.Sha256) { throw 'The external process opened different bytes' }
    Write-Output 'The external process opened the verified package at the exact path returned by the updater.'
} finally {
    Remove-AppxPackage -Package $package.PackageFullName
}
