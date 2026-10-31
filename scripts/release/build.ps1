# Builds one Windows release flavor into a zip (docs/RELEASE.md), the Windows twin of build.sh: dromos-FLAVOR.zip, which
# unpacks with no top folder to bin\dromos.exe and bin\dromos_cpu.dll (with dromos_kernels.dll and the CUDA runtime's
# cudart64_N.dll and cublasLt64_N.dll for the CUDA flavors), share\dromos\tuning, share\dromos\channel (tarball), LICENSE,
# NOTICE and share\dromos\licenses (with NVIDIA's EULA for the CUDA runtime).
#
#   scripts\release\build.ps1 FLAVOR OUTDIR
#
# FLAVOR is windows-x86_64-cpu, windows-x86_64-cuda12 or windows-x86_64-cuda13. Run it on Windows with Visual Studio
# 2022 or its Build Tools (the C++ workload, which brings CMake and Ninja) and Python 3, and for the CUDA flavors the
# flavor's toolkit under CUDA_HOME (scripts\release\cuda-toolkit.ps1 12.8 puts CUDA 12.8 in C:\cuda\12.8 for cuda12,
# and 13.4 puts CUDA 13.4 in C:\cuda\13.4 for cuda13). The MSVC environment (vcvars64) is loaded here when
# cl is not on PATH, so a plain PowerShell will do. The binaries link the C runtime statically: the build refuses one
# that would need the Visual C++ redistributable on the user's machine.
#
# Environment, as for build.sh:
#   LX_VERSION        the version to report, X.Y.Z (default: the CMake rule, git describe --tags --match 'v*')
#   LX_GIT_SHA        the commit's short id, when the checkout has no git (default: git rev-parse)
#   LX_JOBS           build parallelism (default 2)
#   LX_ALLOW_DIRTY    1 to build an uncommitted tree (refused by default, so a release is a commit)
#   CUDA_HOME         the CUDA toolkit (default C:\cuda\12.8 for cuda12 and C:\cuda\13.4 for cuda13, else CUDA_PATH)
#   SOURCE_DATE_EPOCH the zip's timestamps (default: the commit time, or now)
param(
  [Parameter(Mandatory = $true, Position = 0)][string]$Flavor,
  [Parameter(Mandatory = $true, Position = 1)][string]$OutDir
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Fail([string]$message) { throw "build.ps1: $message" }

# Runs a native command and stops on a non-zero exit, as `set -e` does for build.sh. What it writes to stderr is not an
# error by itself (Windows PowerShell would stop on a CMake warning otherwise).
function Invoke-Checked {
  param([string]$File, [string[]]$Arguments)
  $ErrorActionPreference = "Continue"
  & $File @Arguments
  if ($LASTEXITCODE -ne 0) { Fail "$File $($Arguments -join ' ') exited with $LASTEXITCODE" }
}

# git's answer in the source tree, or nothing when it is not a checkout.
function Invoke-Git {
  $ErrorActionPreference = "Continue"
  $out = & git -C $src @args 2>$null
  if ($LASTEXITCODE -eq 0) { $out }
}

$src = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$jobs = if ($env:LX_JOBS) { $env:LX_JOBS } else { "2" }

# The MSVC x64 environment, from the newest Visual Studio or Build Tools with the C++ tools.
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { Fail "no Visual Studio found (vswhere.exe is missing): install the Build Tools' C++ workload" }
  $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
  if (-not $vs) { Fail "no Visual Studio with the x64 C++ tools" }
  cmd /c "`"$vs\VC\Auxiliary\Build\vcvars64.bat`" >nul && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item "env:$($matches[1])" $matches[2] }
  }
  foreach ($tool in @("Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin", "Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja")) {
    if (Test-Path "$vs\$tool") { $env:Path = "$vs\$tool;$env:Path" }
  }
}
foreach ($tool in @("cl.exe", "cmake.exe", "ninja.exe", "dumpbin.exe")) {
  if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) { Fail "$tool is not on PATH after loading the MSVC environment" }
}
$python = (Get-Command python.exe -ErrorAction SilentlyContinue)
if (-not $python) { $python = (Get-Command py.exe -ErrorAction SilentlyContinue) }
if (-not $python) { Fail "Python 3 is needed (the dispatch table and the zip are made with it)" }
$python = $python.Source

# The arguments and targets each flavor is built with (see build.sh for why each one is there).
$cmakeArgs = @("-DLX_FLAVOR=$Flavor", "-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CXX_SCAN_FOR_MODULES=OFF",
               "-DPython3_EXECUTABLE=$python")
$targets = @("dromos")
switch ($Flavor) {
  { $_ -in @("windows-x86_64-cuda12", "windows-x86_64-cuda13") } {
    $series = if ($Flavor -eq "windows-x86_64-cuda13") { "13.4" } else { "12.8" }
    $cuda = if ($env:CUDA_HOME) { $env:CUDA_HOME } elseif (Test-Path "C:\cuda\$series\bin\nvcc.exe") { "C:\cuda\$series" } else { $env:CUDA_PATH }
    if (-not $cuda -or -not (Test-Path "$cuda\bin\nvcc.exe")) { Fail "no CUDA toolkit: set CUDA_HOME (scripts\release\cuda-toolkit.ps1 $series installs one)" }
    $cmakeArgs += @("-DLX_BACKEND_CUDA=ON", "-DLX_BACKEND_CPU=ON", "-DLX_BUNDLE_CUDA=ON", "-DLX_ARCHS=all",
                    "-DLX_LINEINFO=OFF", "-DLX_DECODE_PROGRAM=OFF", "-DCUDAToolkit_ROOT=$cuda",
                    "-DCMAKE_CUDA_COMPILER=$cuda\bin\nvcc.exe")
    $targets += @("dromos_kernels", "dromos_cpu")
  }
  "windows-x86_64-cpu" {
    $cmakeArgs += @("-DLX_BACKEND_CUDA=OFF", "-DLX_BACKEND_CPU=ON")
    $targets += @("dromos_cpu")
  }
  default { Fail "unknown flavor $Flavor (windows-x86_64-cpu, windows-x86_64-cuda12 or windows-x86_64-cuda13)" }
}
if ($env:LX_VERSION) { $cmakeArgs += "-DLX_VERSION=$($env:LX_VERSION)" }

# A release is a commit: refuse a checkout with uncommitted changes, which would not be the tag.
$isRepo = [bool](Get-Command git.exe -ErrorAction SilentlyContinue) -and [bool](Invoke-Git rev-parse --git-dir)
if ($isRepo -and $env:LX_ALLOW_DIRTY -ne "1") {
  if (Invoke-Git status --porcelain) { Fail "the checkout has uncommitted changes; commit them or set LX_ALLOW_DIRTY=1" }
}

New-Item -ItemType Directory -Force $OutDir | Out-Null
$out = (Resolve-Path $OutDir).Path
$build = Join-Path $out "build-$Flavor"
$stage = Join-Path $out "stage-$Flavor"
foreach ($dir in @($build, $stage)) { if (Test-Path $dir) { Remove-Item -Recurse -Force $dir } }

# Clean configure and build of the flavor's own targets, then the install tree.
Invoke-Checked cmake (@("-S", $src, "-B", $build, "-G", "Ninja") + $cmakeArgs)
Invoke-Checked cmake (@("--build", $build, "--target") + $targets + @("-j", $jobs))
Invoke-Checked cmake @("--install", $build, "--prefix", $stage)

# The packager says how this install was made (docs/RELEASE.md: share\dromos\channel). The release archive is the
# "tarball" channel on every platform: `dromos update` replaces it.
New-Item -ItemType Directory -Force (Join-Path $stage "share\dromos") | Out-Null
[IO.File]::WriteAllText((Join-Path $stage "share\dromos\channel"), "tarball`n")

# The tree must be whole before it is packed.
$required = @("bin\dromos.exe", "bin\dromos_cpu.dll", "share\dromos\channel", "LICENSE", "NOTICE",
              "share\dromos\licenses\cpp-httplib.txt")
if ($Flavor -like "*cuda*") {
  $required += @("bin\dromos_kernels.dll", "share\dromos\licenses\NVIDIA-CUDA-EULA.txt")
  foreach ($pattern in @("cudart64_*.dll", "cublasLt64_*.dll")) {
    if (-not (Get-ChildItem (Join-Path $stage "bin") -Filter $pattern)) { Fail "no $pattern beside the CUDA backend" }
  }
}
foreach ($f in $required) { if (-not (Test-Path (Join-Path $stage $f))) { Fail "the install tree has no $f" } }
if (-not (Get-ChildItem (Join-Path $stage "share\dromos\tuning") -Filter *.json -ErrorAction SilentlyContinue)) {
  Fail "no tuning tables in the install tree"
}

# Our binaries must run on a Windows with nothing installed: no Visual C++ redistributable, no debug runtime.
$ours = Get-ChildItem (Join-Path $stage "bin") | Where-Object { $_.Name -eq "dromos.exe" -or $_.Name -like "dromos_*.dll" }
foreach ($binary in $ours) {
  $deps = Invoke-Checked dumpbin.exe @("/nologo", "/dependents", $binary.FullName) | Where-Object { $_ -match '^\s+\S+\.dll\s*$' } | ForEach-Object { $_.Trim() }
  $runtime = $deps | Where-Object { $_ -match '^(vcruntime|msvcp|concrt|vcomp|ucrtbased|api-ms-win-crt-)' }
  if ($runtime) { Fail "$($binary.Name) needs $($runtime -join ', '): the C runtime must be linked statically" }
  Write-Host ("build.ps1: {0} links {1}" -f $binary.Name, ($deps -join ", "))
}

# The tree's own dromos reports the release it is, and loads no backend to say so.
$expectVersion = if ($env:LX_VERSION) { $env:LX_VERSION } else { "" }
$printed = & (Join-Path $stage "bin\dromos.exe") --version
if ($LASTEXITCODE -ne 0 -or $printed -notmatch "^dromos \S+ \($([regex]::Escape($Flavor)), ") { Fail "bin\dromos.exe --version printed: $printed" }
if ($expectVersion -and $printed -notmatch "^dromos $([regex]::Escape($expectVersion)) ") { Fail "bin\dromos.exe is not $expectVersion ($printed)" }
Write-Host "build.ps1: $printed"

# Packed with no top folder and fixed timestamps, so the same tree gives the same bytes.
$epoch = $env:SOURCE_DATE_EPOCH
if (-not $epoch -and $isRepo) { $epoch = Invoke-Git log -1 --format=%ct }
if (-not $epoch) { $epoch = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() }
$zip = Join-Path $out "dromos-$Flavor.zip"
Invoke-Checked $python @((Join-Path $src "scripts\release\pack_zip.py"), $stage, $zip, "$epoch")
$size = "{0:N1} MB" -f ((Get-Item $zip).Length / 1MB)
Write-Host "build.ps1: $zip $size ($((Get-ChildItem (Join-Path $stage 'bin')).Count) files in bin\)"
