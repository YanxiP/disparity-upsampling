param(
    [string]$OpenCVDir = '',
    [string]$BuildDir = 'build_native'
)

$ErrorActionPreference = 'Stop'
$projectRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
if (-not $OpenCVDir) {
    $OpenCVDir = $env:OpenCV_DIR
}
if (-not $OpenCVDir) { throw 'Pass -OpenCVDir or set OpenCV_DIR to an OpenCV C++ build directory.' }
$OpenCVDir = [System.IO.Path]::GetFullPath($OpenCVDir)
if (-not (Test-Path -LiteralPath (Join-Path $OpenCVDir 'OpenCVConfig.cmake'))) {
    throw "OpenCV C++ development package not found: $OpenCVDir"
}

$cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
if ($cmakeCmd) {
    $cmakeExe = $cmakeCmd.Source
} elseif (Test-Path (Join-Path $projectRoot '.vendor/cmake/data/bin/cmake.exe')) {
    $cmakeExe = Join-Path $projectRoot '.vendor/cmake/data/bin/cmake.exe'
} else {
    throw 'CMake not found. Install CMake or add it to PATH.'
}

# MinGW ld cannot open archives through this project's non-ASCII directory.
# Map a temporary ASCII drive for the link step, then remove it.
$drive = @('X','Y','Z') | Where-Object { -not (Test-Path "${_}:\") } | Select-Object -First 1
if (-not $drive) { throw 'No free X/Y/Z drive letter for the MinGW path workaround.' }
$mapped = "${drive}:"
& subst $mapped $projectRoot
if ($LASTEXITCODE -ne 0) { throw 'subst failed' }
try {
    $buildPath = Join-Path $projectRoot $BuildDir
    & $cmakeExe -S $projectRoot -B $buildPath -G 'MinGW Makefiles' "-DOpenCV_DIR=$OpenCVDir"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
    $oldPrefix = $projectRoot.Replace('\','/') + '/'
    $newPrefix = "${drive}:/"
    Get-ChildItem -LiteralPath (Join-Path $buildPath 'CMakeFiles') -Recurse -Filter 'linkLibs.rsp' | ForEach-Object {
        $content = [System.IO.File]::ReadAllText($_.FullName)
        [System.IO.File]::WriteAllText($_.FullName, $content.Replace($oldPrefix, $newPrefix))
    }
    & $cmakeExe --build $buildPath --parallel 4
    if ($LASTEXITCODE -ne 0) { throw 'Native C++ build failed' }
    Write-Output "Built: $buildPath"
} finally {
    & subst $mapped /D | Out-Null
}
