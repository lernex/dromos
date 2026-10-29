# Installs the CUDA toolkit a Windows CUDA release flavor is built with (docs/RELEASE.md), the Windows twin of
# cuda-toolkit.sh. Only what the engine compiles and links: the compiler, the runtime and its headers, cuBLAS
# (cuBLASLt), CCCL (CUB), and the documentation archive. They come from NVIDIA's redistributable archives, each
# checked against the SHA-256 that NVIDIA's manifest lists. No installer runs, nothing needs administrator rights,
# and it works in a non-interactive session. The NVIDIA installer does not.
#
#   scripts/release/cuda-toolkit.ps1 12.8                   # windows-x86_64-cuda12, into C:\cuda\12.8
#   scripts/release/cuda-toolkit.ps1 12.8 -Dest D:\cuda     # somewhere else
#
# The series' newest manifest is used (redistrib_12.8.1.json for 12.8, today). The toolkit lands in
# DEST\bin\nvcc.exe with include\ and lib\x64\ beside it. That is CUDA_HOME for build.ps1. Each archive's license
# is kept in DEST\licenses. Under GitHub Actions, CUDA_HOME and CUDA_PATH are exported to the following steps.
param(
  [Parameter(Mandatory = $true)][string]$Version,
  [string]$Dest = ""
)
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

if ($Version -notmatch '^\d+\.\d+$') { throw "cuda-toolkit.ps1: the version is MAJOR.MINOR, like 12.8 (got $Version)" }
if (-not $Dest) { $Dest = "C:\cuda\$Version" }
$redist = "https://developer.download.nvidia.com/compute/cuda/redist"
$components = @("cuda_nvcc", "cuda_cudart", "cuda_cccl", "libcublas", "cuda_documentation")

# The newest manifest of the series: redistrib_12.8.1.json beats redistrib_12.8.0.json.
$index = (Invoke-WebRequest -UseBasicParsing -Uri "$redist/").Content
$escaped = [regex]::Escape($Version)
$found = [regex]::Matches($index, "redistrib_($escaped(?:\.\d+)?)\.json") | ForEach-Object { $_.Groups[1].Value } |
  Sort-Object -Unique { [version]($_ + $(if ($_.Split('.').Count -lt 3) { ".0" } else { "" })) }
if (-not $found) { throw "cuda-toolkit.ps1: NVIDIA lists no redistributable manifest for CUDA $Version" }
$release = @($found)[-1]
$manifest = Invoke-RestMethod -UseBasicParsing -Uri "$redist/redistrib_$release.json"
Write-Host "cuda-toolkit.ps1: CUDA $release into $Dest"

$work = Join-Path ([IO.Path]::GetTempPath()) ("cuda-redist-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $work, $Dest, (Join-Path $Dest "licenses") | Out-Null
try {
  foreach ($name in $components) {
    $entry = $manifest.$name
    if (-not $entry) { throw "cuda-toolkit.ps1: the CUDA $release manifest has no $name" }
    $file = $entry.'windows-x86_64'
    if (-not $file) { throw "cuda-toolkit.ps1: $name has no windows-x86_64 archive in CUDA $release" }
    $zip = Join-Path $work ([IO.Path]::GetFileName($file.relative_path))
    Invoke-WebRequest -UseBasicParsing -Uri "$redist/$($file.relative_path)" -OutFile $zip
    $sha = (Get-FileHash -Algorithm SHA256 $zip).Hash.ToLowerInvariant()
    if ($sha -ne $file.sha256) { throw "cuda-toolkit.ps1: $name does not match NVIDIA's SHA-256 (expected $($file.sha256), got $sha)" }
    $unpacked = Join-Path $work $name
    tar.exe -xf $zip -C $work
    if ($LASTEXITCODE -ne 0) { throw "cuda-toolkit.ps1: cannot unpack $zip" }
    $top = Join-Path $work ([IO.Path]::GetFileNameWithoutExtension($zip))
    # Every archive is one folder (cuda_nvcc-windows-x86_64-12.8.93-archive) holding bin, include, lib and LICENSE.
    Get-ChildItem -Force $top | Where-Object { $_.Name -ne "LICENSE" } | ForEach-Object {
      Copy-Item -Recurse -Force $_.FullName $Dest
    }
    if (Test-Path (Join-Path $top "LICENSE")) {
      Copy-Item -Force (Join-Path $top "LICENSE") (Join-Path $Dest "licenses\$name-LICENSE.txt")
    }
    Remove-Item -Force $zip
    Remove-Item -Recurse -Force $top
    Write-Host ("cuda-toolkit.ps1: {0} {1}" -f $name, $entry.version)
  }
} finally {
  Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

$nvcc = Join-Path $Dest "bin\nvcc.exe"
if (-not (Test-Path $nvcc)) { throw "cuda-toolkit.ps1: no nvcc under $Dest" }
# The license the bundled cudart and cuBLASLt ship under. The redistributable archives each carry it as LICENSE.
if (-not (Test-Path (Join-Path $Dest "licenses\cuda_cudart-LICENSE.txt"))) { throw "cuda-toolkit.ps1: no license beside cudart" }
& $nvcc --version | Select-Object -Last 1
if ($env:GITHUB_ENV) {
  "CUDA_HOME=$Dest" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
  "CUDA_PATH=$Dest" | Out-File -Append -Encoding utf8 $env:GITHUB_ENV
}
