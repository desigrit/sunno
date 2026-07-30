# Live Captions launcher.
#   .\run.ps1                      # default mic
#   .\run.ps1 -Device 27           # specific input device
#   .\run.ps1 -ListDevices         # show input devices and exit
#   .\run.ps1 -Lan                 # also serve to other devices on the network
#   .\run.ps1 -NoWindow            # server only (open the UI yourself)

[CmdletBinding()]
param(
  [string]$Device,
  [switch]$ListDevices,
  [switch]$Lan,
  [switch]$NoWindow,
  [string]$Model = "large-v3",
  [int]$HttpPort = 8765,
  [int]$WsPort = 8766
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $python)) {
  Write-Error "Virtual environment not found. See README.md for setup."
}

$env:PYTHONUTF8 = "1"

if ($ListDevices) { & $python -m server.app --list-devices; return }

$serverArgs = @("-m", "server.app", "--model", $Model,
                "--http-port", $HttpPort, "--ws-port", $WsPort)
if ($Device) { $serverArgs += @("--device", $Device) }
if ($Lan)    { $serverArgs += @("--host", "0.0.0.0") }

if ($NoWindow) {
  & $python @serverArgs
  return
}

$server = Start-Process -FilePath $python -ArgumentList $serverArgs `
                        -WorkingDirectory $root -PassThru -NoNewWindow

try {
  # Wait for the UI to come up (model load takes ~30 s on a cold start).
  $url = "http://127.0.0.1:$HttpPort/index.html"
  $ready = $false
  foreach ($i in 1..90) {
    if ($server.HasExited) { throw "Server exited with code $($server.ExitCode)." }
    try {
      Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 | Out-Null
      $ready = $true; break
    } catch { Start-Sleep -Milliseconds 700 }
  }
  if (-not $ready) { throw "Server did not start listening on port $HttpPort." }

  # Open in app mode so there's no browser chrome, then pin it above other windows.
  # ${env:ProgramFiles(x86)} needs braces: $env:ProgramFiles(x86) parses as
  # $env:ProgramFiles followed by a literal "(x86)".
  $browser = @(
    "${env:ProgramFiles(x86)}\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Microsoft\Edge\Application\msedge.exe",
    "$env:ProgramFiles\Google\Chrome\Application\chrome.exe",
    "${env:ProgramFiles(x86)}\Google\Chrome\Application\chrome.exe"
  ) | Where-Object { Test-Path $_ } | Select-Object -First 1

  if ($browser) {
    $profileDir = Join-Path $env:LOCALAPPDATA "LiveCaptions\browser-profile"
    $ui = Start-Process -FilePath $browser -PassThru -ArgumentList @(
      "--app=$url", "--window-size=760,340", "--user-data-dir=`"$profileDir`""
    )

    Add-Type @"
using System;
using System.Runtime.InteropServices;
public static class TopMost {
  [DllImport("user32.dll")]
  public static extern bool SetWindowPos(IntPtr hWnd, IntPtr after, int X, int Y,
                                         int cx, int cy, uint flags);
}
"@
    foreach ($i in 1..40) {
      Start-Sleep -Milliseconds 400
      $ui.Refresh()
      if ($ui.MainWindowHandle -ne 0) {
        # HWND_TOPMOST(-1), SWP_NOMOVE|SWP_NOSIZE|SWP_NOACTIVATE
        [TopMost]::SetWindowPos($ui.MainWindowHandle, [IntPtr]::new(-1), 0, 0, 0, 0, 0x0013) | Out-Null
        break
      }
    }
  } else {
    Start-Process $url
  }

  Write-Host ""
  Write-Host "  Live Captions running. Close this window or press Ctrl+C to stop." -ForegroundColor Green
  if ($Lan) {
    $ip = (Get-NetIPAddress -AddressFamily IPv4 |
           Where-Object { $_.PrefixOrigin -in 'Dhcp','Manual' -and $_.IPAddress -ne '127.0.0.1' } |
           Select-Object -First 1).IPAddress
    Write-Host "  Open from another device: http://${ip}:$HttpPort" -ForegroundColor Cyan
  }
  Write-Host ""

  $server.WaitForExit()
}
finally {
  if ($server -and -not $server.HasExited) { Stop-Process -Id $server.Id -Force }
}
