@echo off

REM Check if nuget.exe exists in the PATH
where nuget.exe > nul
if %errorlevel% equ 0 (
    echo nuget.exe already exists.
) else (
    echo nuget.exe not found. Downloading...
    curl -o nuget.exe https://dist.nuget.org/win-x86-commandline/latest/nuget.exe
    if %errorlevel% equ 0 (
        echo nuget.exe downloaded successfully.
    ) else (
        echo Failed to download nuget.exe.
    )
)

REM download required packages using nuget
echo downloading required packages using nuget
set nuget_packages=%~dp0\..\src\thirdparty\nuget_packages
nuget install "Microsoft.AI.DirectML" -Version 1.12.1 -ExcludeVersion -OutputDirectory %nuget_packages%
nuget install "Microsoft.ML.OnnxRuntime.DirectML" -Version 1.15.1 -ExcludeVersion -OutputDirectory %nuget_packages%
