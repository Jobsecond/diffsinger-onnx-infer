$projectRoot = $PSScriptRoot + "/.."

$nugetExePath = $projectRoot + "/nuget.exe"
$packagesPath = $projectRoot + "/src/thirdparty/nuget_packages"
$packagesFile = $projectRoot + "/nuget_packages.txt"

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

# Read the package names and versions from the file
$packages = Get-Content -Path $packagesFile

# Install each package to the specified directory
$packages | ForEach-Object {
    $packageName = ($_ -split "=")[0]
    $packageVersion = ($_ -split "=")[1]

    # Install the package to the specified directory
    & $nugetExePath install $packageName -Version $packageVersion -OutputDirectory $packagesPath
}
