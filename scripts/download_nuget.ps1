param (
    [Parameter(Position = 0)]
    [string]$OrtBuild = $null,

    [Parameter(Position = 1)]
    [string]$Platform = $null,

    [string]$OutConfig = $null,
    [switch]$Overwrite = $false
)


function Write-HostAndFile {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [string]$FilePath,
        [ConsoleColor]$ForegroundColor = 'White',
        [ConsoleColor]$BackgroundColor = 'Black',
        [switch]$NoNewLine
    )

    $originalForegroundColor = $host.ui.rawui.ForegroundColor
    $originalBackgroundColor = $host.ui.rawui.BackgroundColor

    try {
        $host.ui.rawui.ForegroundColor = $ForegroundColor
        $host.ui.rawui.BackgroundColor = $BackgroundColor

        if ($NoNewLine) {
            Write-Host -NoNewLine $Message
        } else {
            Write-Host $Message
        }

        if ($FilePath) {
            if ($NoNewLine) {
                $Message | Out-File -FilePath $FilePath -Append -NoNewline
            } else {
                $Message | Out-File -FilePath $FilePath -Append
            }
        }
    }
    finally {
        $host.ui.rawui.ForegroundColor = $originalForegroundColor
        $host.ui.rawui.BackgroundColor = $originalBackgroundColor
    }
}

function Get-InverseString {
    param (
        [Parameter(Mandatory = $true)]
        [string]$SourceString
    )

    # Split the source string by dashes
    $sections = $SourceString -split "-"

    # Reverse the sections
    $reversedSections = $sections[-1..-($sections.Count)]

    # Join the reversed sections with dashes
    $inverseString = $reversedSections -join "-"

    return $inverseString
}

# Set paths
$projectRoot = "$PSScriptRoot/.."

$nugetExePath = "$projectRoot/nuget.exe"
$packagesPath = "$projectRoot/src/thirdparty/nuget_packages"

$nugetConfigDirName = "nuget_config"


# Select platform
$Platform = $Platform.Trim()

if ($Platform -ieq $null -Or $Platform -ieq "") {
    $Platform = "win-x64"
}

# x64-win -> win-x64
$PlatformInverse = Get-InverseString -SourceString $Platform

if ($Platform -match "(x64|x86|arm64|arm)-(win|osx|linux)") {
    $Platform, $PlatformInverse = $PlatformInverse, $Platform
}

Write-Host "Selected platform $Platform" -ForegroundColor Cyan

# Determine ONNX Runtime build (cpu, gpu, DirectML)
if ($OrtBuild -match "^\s*(dml|directml)\s*$") {
    if ($Platform -match "^win-.*") {
        Write-Host "Selected DirectML version of ONNX Runtime. DirectML libraries will be downloaded." -ForegroundColor Cyan
        $nugetConfigVariety = "directml"
        $OrtEnableDml = $true
    }
    else {
        Write-Host "DirectML is only supported on Windows. Use CPU version of ONNX Runtime instead." -ForegroundColor Yellow
        $nugetConfigVariety = "cpu"
    }
}
elseif ($OrtBuild -match "^\s*(gpu|cuda)\s*$") {
    Write-Host "Selected GPU (CUDA) version of ONNX Runtime." -ForegroundColor Cyan
    $nugetConfigVariety = "gpu"
    $OrtEnableCUDA = $true
}
else {
    Write-Host "Selected CPU version of ONNX Runtime." -ForegroundColor Cyan
    $nugetConfigVariety = "cpu"
}

$nugetPkgConfigPath = "$PSScriptRoot/$nugetConfigDirName/$nugetConfigVariety/packages.config"

$nugetCachePath = $projectRoot + "/.nuget"

$env:NUGET_PACKAGES = $nugetCachePath

# Handle proxy environment variables
$proxyVariables = Get-ChildItem -Path "Env:" | Where-Object { $_.Name -match "all_proxy|https_proxy|http_proxy" }

$proxyUrl = ""

if ($proxyVariables) {
    foreach ($variable in $proxyVariables) {
        $value = $variable.Value

        if ($value -like "http://*" -or $value -like "https://*") {
            $proxyUrl = $value
            break
        }
    }
}

# Check if nuget.exe exists, download if not
if (-not (Test-Path -Path $nugetExePath)) {
    Write-Output "NuGet executable not found. Downloading it now..."
    $nugetUrl = "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
    if ($proxyUrl) {
        Invoke-WebRequest -Uri $nugetUrl -OutFile $nugetExePath -Proxy $proxyUrl
    }
    else {
        Invoke-WebRequest -Uri $nugetUrl -OutFile $nugetExePath
    }
}

# Install nuget packages to the specified directory
& $nugetExePath install $nugetPkgConfigPath -OutputDirectory $packagesPath

[xml]$xmlObject = Get-Content -Path $nugetPkgConfigPath
$packageNodes = $xmlObject.packages.package

foreach ($packageNode in $packageNodes) {
    $packageName = $packageNode.id
    $packageVersion = $packageNode.version

    if ($packageName -ieq "Microsoft.ML.OnnxRuntime" -or $packageName -ilike "Microsoft.ML.OnnxRuntime.*") {
        $OrtIncludePath = Resolve-Path -Path "$packagesPath/$packageName.$packageVersion/build/native/include"
        $OrtLibPath = Resolve-Path -Path "$packagesPath/$packageName.$packageVersion/runtimes/$Platform/native"
    }

    if ($packageName -ieq "Microsoft.AI.DirectML") {
        $DmlIncludePath = Resolve-Path -Path "$packagesPath/$packageName.$packageVersion/include"
        $DmlLibPath = Resolve-Path -Path "$packagesPath/$packageName.$packageVersion/bin/$PlatformInverse"
    }
}


Write-Host ""
Write-Host "Add these to CMake command-line options:" -ForegroundColor Green

if ($Overwrite) {
    # Clear existing output file
    "" | Out-File -FilePath $OutConfig -NoNewLine
}

if ($OrtIncludePath -ne $null -and $OrtLibPath -ne $null) {
    Write-HostAndFile -Message "-DONNXRUNTIME_INCLUDE_PATH=" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message """$OrtIncludePath""" -ForegroundColor DarkCyan -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message "-DONNXRUNTIME_LIB_PATH=" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message """$OrtLibPath""" -ForegroundColor DarkCyan -NoNewLine -FilePath $OutConfig
    if ($OrtEnableCUDA) {
        Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
        Write-HostAndFile -Message "-DENABLE_CUDA:BOOL=ON" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
    }
}
if ($OrtEnableDml -and $DmlIncludePath -ne $null -and $DmlLibPath -ne $null) {
    Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message "-DDML_INCLUDE_PATH=" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message """$DmlIncludePath""" -ForegroundColor DarkCyan -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message "-DDML_LIB_PATH=" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message """$DmlLibPath""" -ForegroundColor DarkCyan -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
    Write-HostAndFile -Message "-DENABLE_DML:BOOL=ON" -ForegroundColor Yellow -NoNewLine -FilePath $OutConfig
}
Write-HostAndFile -Message " " -NoNewLine -FilePath $OutConfig
Write-Host ""
Write-Host ""
