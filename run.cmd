@echo off
REM Launcher that works regardless of PowerShell's execution policy.
REM Windows PowerShell 5.1 defaults to Restricted, which blocks run.ps1 directly.
REM Usage is identical:  run -ListDevices   |   run -Device 27   |   run -Device 27 -Lan
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
