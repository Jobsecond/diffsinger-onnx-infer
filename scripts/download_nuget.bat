@echo off
powershell.exe -ExecutionPolicy Bypass -File %~dp0\download_nuget.ps1 %*
