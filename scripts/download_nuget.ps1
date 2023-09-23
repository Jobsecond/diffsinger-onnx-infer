$projectRoot = $PSScriptRoot + "/.."

$nugetExePath = $projectRoot + "/nuget.exe"
$packagesPath = $projectRoot + "/src/thirdparty/nuget_packages"
$packagesFile = $projectRoot + "/nuget_packages.txt"

# Check if nuget.exe exists, download if not
if (-not (Test-Path -Path $nugetExePath)) {
    Write-Output "NuGet executable not found. Downloading it now..."
    $nugetUrl = "https://dist.nuget.org/win-x86-commandline/latest/nuget.exe"
    Invoke-WebRequest -Uri $nugetUrl -OutFile $nugetExePath
}

# Read the package names and versions from the file
$packages = Get-Content -Path $packagesFile

# Install each package to the specified directory
$packages | ForEach-Object {
    $packageName = ($_ -split "=")[0]
    $packageVersion = ($_ -split "=")[1]

    # Install the package to the specified directory
    & $nugetExePath install $packageName -Version $packageVersion -ExcludeVersion -OutputDirectory $packagesPath
}
