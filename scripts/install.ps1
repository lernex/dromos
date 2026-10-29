# Installs Lernex, the inference engine for Praxis and Logos, on Windows. The twin of install.sh:
#
#   irm https://github.com/lernex/lernex-engine/releases/latest/download/install.ps1 | iex
#
# It picks the build this PC can run (CUDA 12 when an NVIDIA driver is new enough, else CPU), checks it against
# SHA256SUMS, unpacks it to %LOCALAPPDATA%\Programs\lernex-engine\versions\X.Y.Z, points current at it and puts
# current\bin on your user PATH. Run it again to upgrade. Nothing here needs administrator rights.
#
# Environment:
#   LERNEX_VERSION=X.Y.Z        install this release (default: the latest)
#   LERNEX_FLAVOR=cpu|cuda12    install this build (default: the best one for this PC)
#   LERNEX_RELEASES=URL         the release host, laid out like GitHub's (default: GitHub)
#
# Needs 64-bit Windows 10 build 17763 or newer, and Windows PowerShell 5.1 or PowerShell 7.

# The names the engine goes by: the install folder, the command (bin\lernex.exe) and the release assets, which are
# named after the command. A rename is an edit to these two lines.
$LernexFolderName = 'lernex-engine'
$LernexCommandName = 'lernex'

# The CUDA build's floor: NVIDIA's Windows driver for CUDA 12 minor-version compatibility, and the first GPU's
# compute capability (sm_80 is 80).
$LernexCuda12MinDriver = [version]'527.41'
$LernexCuda12MinSm = 80

$LernexDefaultReleases = 'https://github.com/lernex/lernex-engine/releases'

# Messages. Errors are thrown with a "lernex: " prefix and printed once, in Install-Lernex.
function Show-LernexLine([string]$text) {
  Write-Host "lernex: $text"
}

function Show-LernexWarning([string]$text) {
  Write-Host "lernex: $text" -ForegroundColor Yellow
}

# A path with the local app data folder written as %LOCALAPPDATA%, for messages.
function Get-LernexShortPath([string]$path) {
  $prefix = $env:LOCALAPPDATA
  if ($prefix -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    return '%LOCALAPPDATA%' + $path.Substring($prefix.Length)
  }
  return $path
}

# The install root: versions\X.Y.Z, plus the current and previous junctions.
function Get-LernexRoot {
  return Join-Path $env:LOCALAPPDATA "Programs\$LernexFolderName"
}

# A short random name for a temporary or set-aside folder.
function New-LernexId {
  return [guid]::NewGuid().ToString('N').Substring(0, 12)
}

# Stops on anything but 64-bit Windows 10 build 17763 or newer.
function Test-LernexWindows {
  $arch = $env:PROCESSOR_ARCHITEW6432
  if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
  $osArch = ''
  try { $osArch = [string][Runtime.InteropServices.RuntimeInformation]::OSArchitecture } catch { }
  if ($arch -eq 'ARM64' -or $osArch -eq 'Arm64') {
    throw 'lernex: there is no Windows on Arm build yet, and this PC runs Windows on Arm. The Windows build is x64 only'
  }
  if ($arch -ne 'AMD64') {
    throw "lernex: the Windows build is x64 only, and this PC is $arch"
  }
  $build = [Environment]::OSVersion.Version.Build
  if ($build -lt 17763) {
    throw "lernex: Lernex needs Windows 10 build 17763 or newer (it uses tar.exe and junctions), and this PC is build $build"
  }
}

# The nvidia-smi on this PC: the one on PATH, else the driver's own copies. Empty when there is none.
function Get-LernexNvidiaSmi {
  $found = Get-Command nvidia-smi -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
  if ($found) { return [string]$found.Source }
  $fallbacks = @(
    (Join-Path $env:SystemRoot 'System32\nvidia-smi.exe'),
    'C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe'
  )
  foreach ($path in $fallbacks) {
    if (Test-Path -LiteralPath $path) { return $path }
  }
  return ''
}

# The first line nvidia-smi prints for one query field, or '' when it cannot be read.
function Read-LernexSmi([string]$smi, [string]$field) {
  $ErrorActionPreference = 'Continue'
  $lines = @()
  try {
    $lines = @(& $smi "--query-gpu=$field" '--format=csv,noheader' 2>$null)
  } catch {
    return ''
  }
  if ($LASTEXITCODE -ne 0 -or $lines.Count -eq 0) { return '' }
  return ([string]$lines[0]).Trim()
}

# What the PC's NVIDIA GPU can do. Driver is '' when there is no readable NVIDIA driver. Sm is the first GPU's
# compute capability as sm_XY (8.9 is 89), or 0 when it cannot be read.
function Get-LernexNvidia {
  $info = [pscustomobject]@{ Driver = ''; DriverVersion = $null; Gpu = 'NVIDIA GPU'; Sm = 0 }
  $smi = Get-LernexNvidiaSmi
  if (-not $smi) { return $info }
  $driver = Read-LernexSmi $smi 'driver_version'
  if ($driver -notmatch '^\d+\.\d+') { return $info }
  try {
    $info.DriverVersion = [version]$driver
  } catch {
    return $info
  }
  $info.Driver = $driver
  $name = Read-LernexSmi $smi 'name'
  if ($name) { $info.Gpu = $name }
  $cc = Read-LernexSmi $smi 'compute_cap'
  if ($cc -match '^(\d+)\.(\d+)$') { $info.Sm = ([int]$matches[1]) * 10 + [int]$matches[2] }
  return $info
}

# The SHA-256 that SHA256SUMS lists for a file, in lowercase, or '' when the file is not listed.
function Get-LernexListedHash([string]$sums, [string]$name) {
  foreach ($line in Get-Content -LiteralPath $sums) {
    if ($line -match '^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$' -and $matches[2] -eq $name) {
      return $matches[1].ToLowerInvariant()
    }
  }
  return ''
}

# Decides the flavor the way install.sh does, with Windows numbers. Returns Flavor, Why (one line for the user),
# Asset (the zip's name) and Nvidia (what the GPU reported).
function Select-LernexBuild([string]$sums) {
  $nv = Get-LernexNvidia
  $forced = [string]$env:LERNEX_FLAVOR
  $flavor = ''
  $why = ''
  if ($forced) {
    $flavor = $forced.ToLowerInvariant()
    if ($flavor -ne 'cpu' -and $flavor -ne 'cuda12') {
      throw "lernex: LERNEX_FLAVOR must be cpu or cuda12, not $forced"
    }
    $why = 'LERNEX_FLAVOR asks for it'
  } elseif (-not $nv.Driver) {
    $flavor = 'cpu'
    $why = 'no NVIDIA driver found, so no CUDA'
  } elseif ($nv.Sm -eq 0) {
    $flavor = 'cpu'
    $why = "the NVIDIA driver $($nv.Driver) is here, but nvidia-smi cannot read the GPU's compute capability"
  } elseif ($nv.DriverVersion -lt $LernexCuda12MinDriver) {
    $flavor = 'cpu'
    $why = "driver $($nv.Driver) is older than $LernexCuda12MinDriver, the oldest CUDA build needs"
  } elseif ($nv.Sm -lt $LernexCuda12MinSm) {
    $flavor = 'cpu'
    $why = "$($nv.Gpu) is sm_$($nv.Sm), and the CUDA builds need sm_$LernexCuda12MinSm or newer"
  } elseif (Get-LernexListedHash $sums "$LernexCommandName-windows-x86_64-cuda12.zip") {
    $flavor = 'cuda12'
    $why = "$($nv.Gpu), driver $($nv.Driver)"
  } else {
    $flavor = 'cpu'
    $why = "this release has no CUDA build for driver $($nv.Driver)"
  }
  return [pscustomobject]@{
    Flavor = $flavor
    Why = $why
    Asset = "$LernexCommandName-windows-x86_64-$flavor.zip"
    Nvidia = $nv
  }
}

# Downloads a URL to a file. The progress bar is off while it runs, because PowerShell 5.1's progress bar makes the
# transfer many times slower.
function Get-LernexFile([string]$url, [string]$file) {
  $saved = $ProgressPreference
  $ProgressPreference = 'SilentlyContinue'
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $url -OutFile $file
  } finally {
    $ProgressPreference = $saved
  }
}

# Unpacks a release zip into an empty folder. tar.exe (bsdtar) reads zip and is much faster than Expand-Archive,
# which is the fallback when tar.exe is missing.
function Expand-LernexZip([string]$zip, [string]$dest, [string]$asset) {
  $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
  if (Test-Path -LiteralPath $tar) {
    $ErrorActionPreference = 'Continue'
    $out = @(& $tar -xf $zip -C $dest 2>&1 | ForEach-Object { [string]$_ })
    if ($LASTEXITCODE -ne 0) {
      $reason = ''
      if ($out.Count -gt 0) { $reason = " ($($out[0]))" }
      throw "lernex: cannot unpack $asset$reason"
    }
  } else {
    try {
      Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
    } catch {
      throw "lernex: cannot unpack $asset"
    }
  }
}

# The version a unpacked tree reports from its own bin\lernex.exe --version. That prints one line and loads no
# backend.
function Get-LernexTreeVersion([string]$tree, [string]$asset) {
  $exe = Join-Path $tree "bin\$LernexCommandName.exe"
  $ErrorActionPreference = 'Continue'
  $out = @(& $exe --version 2>&1 | ForEach-Object { [string]$_ })
  $first = ''
  if ($out.Count -gt 0) { $first = $out[0] }
  if ($first -match ('^' + [regex]::Escape($LernexCommandName) + ' (\d+\.\d+\.\d+\S*) \(')) {
    return $matches[1]
  }
  throw "lernex: the $LernexCommandName in $asset does not report a version (it printed: $first)"
}

# The folder a junction points to, or '' when the path is not a junction or does not exist.
function Get-LernexLinkTarget([string]$path) {
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if (-not $item) { return '' }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) { return '' }
  $target = $item.Target
  if ($target -is [array]) { $target = $target[0] }
  return [string]$target
}

# Removes a junction. Directory.Delete without recursion removes the link and leaves what it points to alone.
function Remove-LernexLink([string]$path) {
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if (-not $item) { return }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
    throw "lernex: $(Get-LernexShortPath $path) is a folder, not a link, so the installer will not replace it"
  }
  [IO.Directory]::Delete($path, $false)
}

# True when a running process was started from a file under this folder: a lernex that is serving, for example. The
# rename below cannot tell, because Windows lets the folder of a running program be renamed. Processes this user
# cannot see into are not counted.
function Test-LernexInUse([string]$folder) {
  $prefix = $folder.TrimEnd('\') + '\'
  foreach ($proc in @(Get-Process -ErrorAction SilentlyContinue)) {
    $image = ''
    try { $image = [string]$proc.Path } catch { $image = '' }
    if ($image -and $image.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { return $true }
  }
  return $false
}

# Deletes a version folder, after the caller has checked that nothing runs from it. The folder is renamed out of
# versions first, so a rename that fails (a file held open) leaves it where it was. Returns $false when it stayed.
function Remove-LernexVersion([string]$path, [string]$versions) {
  $trash = Join-Path $versions ('.replaced-' + (New-LernexId))
  try {
    [IO.Directory]::Move($path, $trash)
  } catch {
    return $false
  }
  Remove-Item -LiteralPath $trash -Recurse -Force -ErrorAction SilentlyContinue
  return $true
}

# The PATH entry for current\bin, with %LOCALAPPDATA% left unexpanded in the registry.
function Get-LernexPathEntry {
  return "%LOCALAPPDATA%\Programs\$LernexFolderName\current\bin"
}

# A PATH entry in a form to compare: expanded, without a trailing backslash, in lowercase.
function Get-LernexPathKey([string]$entry) {
  return [Environment]::ExpandEnvironmentVariables($entry.Trim()).TrimEnd('\').ToLowerInvariant()
}

# Puts current\bin on the user's PATH once, as an expandable string so new terminals see it, and on this window's
# PATH too, so lernex works here right away.
function Add-LernexPath {
  $entry = Get-LernexPathEntry
  $key = Get-LernexPathKey $entry
  $regKey = 'HKCU:\Environment'
  $raw = [string](Get-Item -LiteralPath $regKey).GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
  $listed = $false
  foreach ($item in ($raw -split ';')) {
    if ($item.Trim() -and (Get-LernexPathKey $item) -eq $key) { $listed = $true }
  }
  if ($listed) {
    Show-LernexLine "your user PATH already has $entry"
  } else {
    $new = $raw
    if ($new -and -not $new.EndsWith(';')) { $new = $new + ';' }
    $new = $new + $entry
    Set-ItemProperty -Path $regKey -Name Path -Value $new -Type ExpandString
    # Setting and then clearing a user variable makes Windows tell the running programs that the environment changed.
    [Environment]::SetEnvironmentVariable('LERNEX_INSTALL_REFRESH', '1', 'User')
    [Environment]::SetEnvironmentVariable('LERNEX_INSTALL_REFRESH', $null, 'User')
    Show-LernexLine "added $entry to your user PATH: open a new terminal to use lernex"
  }
  $expanded = [Environment]::ExpandEnvironmentVariables($entry)
  $sessionHas = $false
  foreach ($item in ([string]$env:Path -split ';')) {
    if ($item.Trim() -and (Get-LernexPathKey $item) -eq $key) { $sessionHas = $true }
  }
  if (-not $sessionHas) {
    $env:Path = $expanded + ';' + [string]$env:Path
    Show-LernexLine 'this window has it on PATH too, so lernex works here now'
  }
}

function Invoke-LernexInstall {
  $ErrorActionPreference = 'Stop'
  Test-LernexWindows

  $releases = [string]$env:LERNEX_RELEASES
  if (-not $releases) { $releases = $LernexDefaultReleases }
  $releases = $releases.TrimEnd('/')
  $version = [string]$env:LERNEX_VERSION
  if ($version) {
    $version = $version -replace '^[vV]', ''
    $base = "$releases/download/v$version"
  } else {
    $base = "$releases/latest/download"
  }

  $root = Get-LernexRoot
  $versions = Join-Path $root 'versions'
  $currentLink = Join-Path $root 'current'
  $previousLink = Join-Path $root 'previous'
  $tmp = ''
  $stage = ''
  $replaced = ''
  $dest = ''
  $tls = $null
  try {
    if ($PSVersionTable.PSVersion.Major -lt 6) {
      $tls = [Net.ServicePointManager]::SecurityProtocol
      [Net.ServicePointManager]::SecurityProtocol = $tls -bor [Net.SecurityProtocolType]::Tls12
    }
    $tmp = Join-Path $env:TEMP ('lernex-install-' + (New-LernexId))
    New-Item -ItemType Directory -Path $tmp | Out-Null

    $sums = Join-Path $tmp 'SHA256SUMS'
    try {
      Get-LernexFile "$base/SHA256SUMS" $sums
    } catch {
      if ($version) { throw "lernex: there is no release $version at $releases (check LERNEX_VERSION)" }
      throw "lernex: cannot download the release from $base (check your connection, or LERNEX_RELEASES)"
    }

    $build = Select-LernexBuild $sums
    $asset = $build.Asset
    $expected = Get-LernexListedHash $sums $asset
    if (-not $expected) {
      throw "lernex: this release has no $asset, so there is no $($build.Flavor) build for this machine"
    }
    Show-LernexLine "Windows x64: $($build.Flavor) build, $($build.Why)"

    $min = $LernexCuda12MinDriver
    if ($build.Flavor -eq 'cuda12' -and ([string]$env:LERNEX_FLAVOR -ne '')) {
      $nv = $build.Nvidia
      if (-not $nv.Driver -or $nv.DriverVersion -lt $min) {
        $have = 'none'
        if ($nv.Driver) { $have = $nv.Driver }
        Show-LernexWarning "cuda12 needs an NVIDIA driver of $min or newer, and this PC has ${have}: the engine may not start"
      }
    }

    # Already current: the installed tree was made from this asset, byte for byte.
    $currentTarget = Get-LernexLinkTarget $currentLink
    if ($currentTarget -and (Test-Path -LiteralPath (Join-Path $currentTarget '.install'))) {
      $installed = (Get-Content -Raw -LiteralPath (Join-Path $currentTarget '.install')).Trim()
      if ($installed -eq "$asset $expected") {
        Show-LernexLine "lernex $(Split-Path -Leaf $currentTarget) is already installed and current"
        Add-LernexPath
        Show-LernexLine "done. Try: $LernexCommandName doctor"
        return
      }
    }

    Show-LernexLine "downloading $asset"
    $zip = Join-Path $tmp $asset
    try {
      Get-LernexFile "$base/$asset" $zip
    } catch {
      throw "lernex: cannot download $base/$asset"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
      throw "lernex: $asset does not match SHA256SUMS (expected $expected, got $actual). Nothing was installed; try again"
    }

    New-Item -ItemType Directory -Force -Path $versions | Out-Null
    $stage = Join-Path $versions ('.incoming-' + (New-LernexId))
    New-Item -ItemType Directory -Path $stage | Out-Null
    Expand-LernexZip $zip $stage $asset
    if (-not (Test-Path -LiteralPath (Join-Path $stage "bin\$LernexCommandName.exe"))) {
      throw "lernex: $asset has no bin\$LernexCommandName.exe, so it is not a Lernex build"
    }
    $ver = Get-LernexTreeVersion $stage $asset
    Set-Content -LiteralPath (Join-Path $stage '.install') -Value "$asset $expected" -Encoding ASCII

    # A folder of the same version is set aside before the new one goes in. A lernex running from it stops the install
    # here, before anything moves.
    $dest = Join-Path $versions $ver
    if (Test-Path -LiteralPath $dest) {
      if (Test-LernexInUse $dest) {
        throw "lernex: lernex $ver is running (close it, for example a '$LernexCommandName serve'), then run the installer again"
      }
      $aside = Join-Path $versions ('.replaced-' + (New-LernexId))
      try {
        [IO.Directory]::Move($dest, $aside)
      } catch {
        throw "lernex: lernex $ver is running (close it, for example a '$LernexCommandName serve'), then run the installer again"
      }
      $replaced = $aside
    }
    try {
      [IO.Directory]::Move($stage, $dest)
    } catch {
      if ($replaced) {
        [IO.Directory]::Move($replaced, $dest)
        $replaced = ''
      }
      throw "lernex: cannot put lernex $ver in $(Get-LernexShortPath $dest)"
    }
    $stage = ''

    # Switch, then tidy. The version this one replaces stays for one rollback, and so does the one before it when this
    # is a re-install of the current version; everything older goes. The junctions are absolute, as install.sh's links are.
    $was = ''
    $wasTarget = Get-LernexLinkTarget $currentLink
    if ($wasTarget) { $was = Split-Path -Leaf $wasTarget }
    $previous = $was
    if ($was -eq $ver) {
      $oldPrevious = Get-LernexLinkTarget $previousLink
      $previous = ''
      if ($oldPrevious) { $previous = Split-Path -Leaf $oldPrevious }
    }
    Remove-LernexLink $currentLink
    New-Item -ItemType Junction -Path $currentLink -Target $dest | Out-Null

    $inUse = @()
    foreach ($dir in @(Get-ChildItem -LiteralPath $versions -Directory -Force)) {
      if ($dir.Name.StartsWith('.')) { continue }
      if ($dir.Name -eq $ver -or $dir.Name -eq $previous) { continue }
      if (Test-LernexInUse $dir.FullName) { $inUse += $dir.Name; continue }
      if (-not (Remove-LernexVersion $dir.FullName $versions)) { $inUse += $dir.Name }
    }
    if ($inUse.Count -gt 0) {
      Show-LernexLine "kept $($inUse -join ', ') for now, because a running lernex is using it. The next install removes it"
    }
    # Set-aside and half-unpacked folders from earlier runs (a lernex was still running from one) go once it has stopped,
    # and so do the ones `lernex update` set aside (.retired-*).
    $leftovers = Get-ChildItem -LiteralPath $versions -Directory -Force |
      Where-Object { $_.Name -like '.replaced-*' -or $_.Name -like '.incoming-*' -or $_.Name -like '.retired-*' }
    foreach ($leftover in @($leftovers)) {
      if (-not (Test-LernexInUse $leftover.FullName)) {
        Remove-Item -LiteralPath $leftover.FullName -Recurse -Force -ErrorAction SilentlyContinue
      }
    }

    Remove-LernexLink $previousLink
    if ($previous -and $previous -ne $ver -and (Test-Path -LiteralPath (Join-Path $versions $previous))) {
      New-Item -ItemType Junction -Path $previousLink -Target (Join-Path $versions $previous) | Out-Null
    }

    $shown = Get-LernexShortPath $dest
    if ($was -and $was -ne $ver) {
      Show-LernexLine "installed lernex $ver in $shown (was $was)"
    } else {
      Show-LernexLine "installed lernex $ver in $shown"
    }
    Add-LernexPath
    Show-LernexLine "done. Try: $LernexCommandName doctor"
  } finally {
    if ($null -ne $tls) { [Net.ServicePointManager]::SecurityProtocol = $tls }
    if ($tmp) { Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue }
    if ($stage) { Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue }
    # The set-aside copy goes only once the new tree is in place; otherwise it is the old tree, and it stays.
    if ($replaced -and $dest -and (Test-Path -LiteralPath $dest)) {
      Remove-Item -LiteralPath $replaced -Recurse -Force -ErrorAction SilentlyContinue
    }
  }
}

# The entry point. An error comes out here once, in red. Under `irm | iex` it runs in the user's own session (or
# their script), so it never calls exit there; run as a file (powershell -File install.ps1, or .\install.ps1) a
# failure exits with 1. A function knows the file it was defined in, and one defined by iex has none.
function Install-Lernex {
  try {
    $null = Invoke-LernexInstall
    return $true
  } catch {
    $message = [string]$_.Exception.Message
    if ($message -notlike 'lernex:*') { $message = "lernex: $message" }
    Write-Host $message -ForegroundColor Red
    return $false
  }
}

$LernexRanAsFile = [bool](Get-Command Install-Lernex -CommandType Function).ScriptBlock.File
if (-not (Install-Lernex) -and $LernexRanAsFile) { exit 1 }
