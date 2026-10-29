# Installs Dromos, the inference engine for Praxis and Logos, on Windows. The twin of install.sh:
#
#   irm https://github.com/lernex/dromos/releases/latest/download/install.ps1 | iex
#
# It picks the build this PC can run (CUDA 12 when an NVIDIA driver is new enough, else CPU), checks it against
# SHA256SUMS, unpacks it to %LOCALAPPDATA%\Programs\Dromos\versions\X.Y.Z, points current at it and puts
# current\bin on your user PATH. Run it again to upgrade. Nothing here needs administrator rights.
#
# Environment:
#   DROMOS_VERSION=X.Y.Z        install this release (default: the latest)
#   DROMOS_FLAVOR=cpu|cuda12    install this build (default: the best one for this PC)
#   DROMOS_RELEASES=URL         the release host, laid out like GitHub's (default: GitHub)
#
# Needs 64-bit Windows 10 build 17763 or newer, and Windows PowerShell 5.1 or PowerShell 7.

# The names the engine goes by: the install folder, the command (bin\dromos.exe) and the release assets, which are
# named after the command. A rename is an edit to these two lines.
$DromosFolderName = 'Dromos'
$DromosCommandName = 'dromos'

# The CUDA build's floor: NVIDIA's Windows driver for CUDA 12 minor-version compatibility, and the first GPU's
# compute capability (sm_80 is 80).
$DromosCuda12MinDriver = [version]'527.41'
$DromosCuda12MinSm = 80

$DromosDefaultReleases = 'https://github.com/lernex/dromos/releases'

# Messages. Errors are thrown with a "dromos: " prefix and printed once, in Install-Dromos.
function Show-DromosLine([string]$text) {
  Write-Host "dromos: $text"
}

function Show-DromosWarning([string]$text) {
  Write-Host "dromos: $text" -ForegroundColor Yellow
}

# A path with the local app data folder written as %LOCALAPPDATA%, for messages.
function Get-DromosShortPath([string]$path) {
  $prefix = $env:LOCALAPPDATA
  if ($prefix -and $path.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
    return '%LOCALAPPDATA%' + $path.Substring($prefix.Length)
  }
  return $path
}

# The install root: versions\X.Y.Z, plus the current and previous junctions.
function Get-DromosRoot {
  return Join-Path $env:LOCALAPPDATA "Programs\$DromosFolderName"
}

# A short random name for a temporary or set-aside folder.
function New-DromosId {
  return [guid]::NewGuid().ToString('N').Substring(0, 12)
}

# Stops on anything but 64-bit Windows 10 build 17763 or newer.
function Test-DromosWindows {
  $arch = $env:PROCESSOR_ARCHITEW6432
  if (-not $arch) { $arch = $env:PROCESSOR_ARCHITECTURE }
  $osArch = ''
  try { $osArch = [string][Runtime.InteropServices.RuntimeInformation]::OSArchitecture } catch { }
  if ($arch -eq 'ARM64' -or $osArch -eq 'Arm64') {
    throw 'dromos: there is no Windows on Arm build yet, and this PC runs Windows on Arm. The Windows build is x64 only'
  }
  if ($arch -ne 'AMD64') {
    throw "dromos: the Windows build is x64 only, and this PC is $arch"
  }
  $build = [Environment]::OSVersion.Version.Build
  if ($build -lt 17763) {
    throw "dromos: Dromos needs Windows 10 build 17763 or newer (it uses tar.exe and junctions), and this PC is build $build"
  }
}

# The nvidia-smi on this PC: the one on PATH, else the driver's own copies. Empty when there is none.
function Get-DromosNvidiaSmi {
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
function Read-DromosSmi([string]$smi, [string]$field) {
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
function Get-DromosNvidia {
  $info = [pscustomobject]@{ Driver = ''; DriverVersion = $null; Gpu = 'NVIDIA GPU'; Sm = 0 }
  $smi = Get-DromosNvidiaSmi
  if (-not $smi) { return $info }
  $driver = Read-DromosSmi $smi 'driver_version'
  if ($driver -notmatch '^\d+\.\d+') { return $info }
  try {
    $info.DriverVersion = [version]$driver
  } catch {
    return $info
  }
  $info.Driver = $driver
  $name = Read-DromosSmi $smi 'name'
  if ($name) { $info.Gpu = $name }
  $cc = Read-DromosSmi $smi 'compute_cap'
  if ($cc -match '^(\d+)\.(\d+)$') { $info.Sm = ([int]$matches[1]) * 10 + [int]$matches[2] }
  return $info
}

# The SHA-256 that SHA256SUMS lists for a file, in lowercase, or '' when the file is not listed.
function Get-DromosListedHash([string]$sums, [string]$name) {
  foreach ($line in Get-Content -LiteralPath $sums) {
    if ($line -match '^([0-9A-Fa-f]{64})\s+\*?(.+?)\s*$' -and $matches[2] -eq $name) {
      return $matches[1].ToLowerInvariant()
    }
  }
  return ''
}

# Decides the flavor the way install.sh does, with Windows numbers. Returns Flavor, Why (one line for the user),
# Asset (the zip's name) and Nvidia (what the GPU reported).
function Select-DromosBuild([string]$sums) {
  $nv = Get-DromosNvidia
  $forced = [string]$env:DROMOS_FLAVOR
  $flavor = ''
  $why = ''
  if ($forced) {
    $flavor = $forced.ToLowerInvariant()
    if ($flavor -ne 'cpu' -and $flavor -ne 'cuda12') {
      throw "dromos: DROMOS_FLAVOR must be cpu or cuda12, not $forced"
    }
    $why = 'DROMOS_FLAVOR asks for it'
  } elseif (-not $nv.Driver) {
    $flavor = 'cpu'
    $why = 'no NVIDIA driver found, so no CUDA'
  } elseif ($nv.Sm -eq 0) {
    $flavor = 'cpu'
    $why = "the NVIDIA driver $($nv.Driver) is here, but nvidia-smi cannot read the GPU's compute capability"
  } elseif ($nv.DriverVersion -lt $DromosCuda12MinDriver) {
    $flavor = 'cpu'
    $why = "driver $($nv.Driver) is older than $DromosCuda12MinDriver, the oldest CUDA build needs"
  } elseif ($nv.Sm -lt $DromosCuda12MinSm) {
    $flavor = 'cpu'
    $why = "$($nv.Gpu) is sm_$($nv.Sm), and the CUDA builds need sm_$DromosCuda12MinSm or newer"
  } elseif (Get-DromosListedHash $sums "$DromosCommandName-windows-x86_64-cuda12.zip") {
    $flavor = 'cuda12'
    $why = "$($nv.Gpu), driver $($nv.Driver)"
  } else {
    $flavor = 'cpu'
    $why = "this release has no CUDA build for driver $($nv.Driver)"
  }
  return [pscustomobject]@{
    Flavor = $flavor
    Why = $why
    Asset = "$DromosCommandName-windows-x86_64-$flavor.zip"
    Nvidia = $nv
  }
}

# Downloads a URL to a file. The progress bar is off while it runs, because PowerShell 5.1's progress bar makes the
# transfer many times slower.
function Get-DromosFile([string]$url, [string]$file) {
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
function Expand-DromosZip([string]$zip, [string]$dest, [string]$asset) {
  $tar = Join-Path $env:SystemRoot 'System32\tar.exe'
  if (Test-Path -LiteralPath $tar) {
    $ErrorActionPreference = 'Continue'
    $out = @(& $tar -xf $zip -C $dest 2>&1 | ForEach-Object { [string]$_ })
    if ($LASTEXITCODE -ne 0) {
      $reason = ''
      if ($out.Count -gt 0) { $reason = " ($($out[0]))" }
      throw "dromos: cannot unpack $asset$reason"
    }
  } else {
    try {
      Expand-Archive -LiteralPath $zip -DestinationPath $dest -Force
    } catch {
      throw "dromos: cannot unpack $asset"
    }
  }
}

# The version a unpacked tree reports from its own bin\dromos.exe --version. That prints one line and loads no
# backend.
function Get-DromosTreeVersion([string]$tree, [string]$asset) {
  $exe = Join-Path $tree "bin\$DromosCommandName.exe"
  $ErrorActionPreference = 'Continue'
  $out = @(& $exe --version 2>&1 | ForEach-Object { [string]$_ })
  $first = ''
  if ($out.Count -gt 0) { $first = $out[0] }
  if ($first -match ('^' + [regex]::Escape($DromosCommandName) + ' (\d+\.\d+\.\d+\S*) \(')) {
    return $matches[1]
  }
  throw "dromos: the $DromosCommandName in $asset does not report a version (it printed: $first)"
}

# The folder a junction points to, or '' when the path is not a junction or does not exist.
function Get-DromosLinkTarget([string]$path) {
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if (-not $item) { return '' }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) { return '' }
  $target = $item.Target
  if ($target -is [array]) { $target = $target[0] }
  return [string]$target
}

# Removes a junction. Directory.Delete without recursion removes the link and leaves what it points to alone.
function Remove-DromosLink([string]$path) {
  $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
  if (-not $item) { return }
  if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0) {
    throw "dromos: $(Get-DromosShortPath $path) is a folder, not a link, so the installer will not replace it"
  }
  [IO.Directory]::Delete($path, $false)
}

# True when a running process was started from a file under this folder: a dromos that is serving, for example. The
# rename below cannot tell, because Windows lets the folder of a running program be renamed. Processes this user
# cannot see into are not counted.
function Test-DromosInUse([string]$folder) {
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
function Remove-DromosVersion([string]$path, [string]$versions) {
  $trash = Join-Path $versions ('.replaced-' + (New-DromosId))
  try {
    [IO.Directory]::Move($path, $trash)
  } catch {
    return $false
  }
  Remove-Item -LiteralPath $trash -Recurse -Force -ErrorAction SilentlyContinue
  return $true
}

# The PATH entry for current\bin, with %LOCALAPPDATA% left unexpanded in the registry.
function Get-DromosPathEntry {
  return "%LOCALAPPDATA%\Programs\$DromosFolderName\current\bin"
}

# A PATH entry in a form to compare: expanded, without a trailing backslash, in lowercase.
function Get-DromosPathKey([string]$entry) {
  return [Environment]::ExpandEnvironmentVariables($entry.Trim()).TrimEnd('\').ToLowerInvariant()
}

# Puts current\bin on the user's PATH once, as an expandable string so new terminals see it, and on this window's
# PATH too, so dromos works here right away.
function Add-DromosPath {
  $entry = Get-DromosPathEntry
  $key = Get-DromosPathKey $entry
  $regKey = 'HKCU:\Environment'
  $raw = [string](Get-Item -LiteralPath $regKey).GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
  $listed = $false
  foreach ($item in ($raw -split ';')) {
    if ($item.Trim() -and (Get-DromosPathKey $item) -eq $key) { $listed = $true }
  }
  if ($listed) {
    Show-DromosLine "your user PATH already has $entry"
  } else {
    $new = $raw
    if ($new -and -not $new.EndsWith(';')) { $new = $new + ';' }
    $new = $new + $entry
    Set-ItemProperty -Path $regKey -Name Path -Value $new -Type ExpandString
    # Setting and then clearing a user variable makes Windows tell the running programs that the environment changed.
    [Environment]::SetEnvironmentVariable('DROMOS_INSTALL_REFRESH', '1', 'User')
    [Environment]::SetEnvironmentVariable('DROMOS_INSTALL_REFRESH', $null, 'User')
    Show-DromosLine "added $entry to your user PATH: open a new terminal to use dromos"
  }
  $expanded = [Environment]::ExpandEnvironmentVariables($entry)
  $sessionHas = $false
  foreach ($item in ([string]$env:Path -split ';')) {
    if ($item.Trim() -and (Get-DromosPathKey $item) -eq $key) { $sessionHas = $true }
  }
  if (-not $sessionHas) {
    $env:Path = $expanded + ';' + [string]$env:Path
    Show-DromosLine 'this window has it on PATH too, so dromos works here now'
  }
}

function Invoke-DromosInstall {
  $ErrorActionPreference = 'Stop'
  Test-DromosWindows

  $releases = [string]$env:DROMOS_RELEASES
  if (-not $releases) { $releases = $DromosDefaultReleases }
  $releases = $releases.TrimEnd('/')
  $version = [string]$env:DROMOS_VERSION
  if ($version) {
    $version = $version -replace '^[vV]', ''
    $base = "$releases/download/v$version"
  } else {
    $base = "$releases/latest/download"
  }

  $root = Get-DromosRoot
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
    $tmp = Join-Path $env:TEMP ('dromos-install-' + (New-DromosId))
    New-Item -ItemType Directory -Path $tmp | Out-Null

    $sums = Join-Path $tmp 'SHA256SUMS'
    try {
      Get-DromosFile "$base/SHA256SUMS" $sums
    } catch {
      if ($version) { throw "dromos: there is no release $version at $releases (check DROMOS_VERSION)" }
      throw "dromos: cannot download the release from $base (check your connection, or DROMOS_RELEASES)"
    }

    $build = Select-DromosBuild $sums
    $asset = $build.Asset
    $expected = Get-DromosListedHash $sums $asset
    if (-not $expected) {
      throw "dromos: this release has no $asset, so there is no $($build.Flavor) build for this machine"
    }
    Show-DromosLine "Windows x64: $($build.Flavor) build, $($build.Why)"

    $min = $DromosCuda12MinDriver
    if ($build.Flavor -eq 'cuda12' -and ([string]$env:DROMOS_FLAVOR -ne '')) {
      $nv = $build.Nvidia
      if (-not $nv.Driver -or $nv.DriverVersion -lt $min) {
        $have = 'none'
        if ($nv.Driver) { $have = $nv.Driver }
        Show-DromosWarning "cuda12 needs an NVIDIA driver of $min or newer, and this PC has ${have}: the engine may not start"
      }
    }

    # Already current: the installed tree was made from this asset, byte for byte.
    $currentTarget = Get-DromosLinkTarget $currentLink
    if ($currentTarget -and (Test-Path -LiteralPath (Join-Path $currentTarget '.install'))) {
      $installed = (Get-Content -Raw -LiteralPath (Join-Path $currentTarget '.install')).Trim()
      if ($installed -eq "$asset $expected") {
        Show-DromosLine "dromos $(Split-Path -Leaf $currentTarget) is already installed and current"
        Add-DromosPath
        Show-DromosLine "done. Try: $DromosCommandName doctor"
        return
      }
    }

    Show-DromosLine "downloading $asset"
    $zip = Join-Path $tmp $asset
    try {
      Get-DromosFile "$base/$asset" $zip
    } catch {
      throw "dromos: cannot download $base/$asset"
    }
    $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $zip).Hash.ToLowerInvariant()
    if ($actual -ne $expected) {
      throw "dromos: $asset does not match SHA256SUMS (expected $expected, got $actual). Nothing was installed; try again"
    }

    New-Item -ItemType Directory -Force -Path $versions | Out-Null
    $stage = Join-Path $versions ('.incoming-' + (New-DromosId))
    New-Item -ItemType Directory -Path $stage | Out-Null
    Expand-DromosZip $zip $stage $asset
    if (-not (Test-Path -LiteralPath (Join-Path $stage "bin\$DromosCommandName.exe"))) {
      throw "dromos: $asset has no bin\$DromosCommandName.exe, so it is not a Dromos build"
    }
    $ver = Get-DromosTreeVersion $stage $asset
    Set-Content -LiteralPath (Join-Path $stage '.install') -Value "$asset $expected" -Encoding ASCII

    # A folder of the same version is set aside before the new one goes in. A dromos running from it stops the install
    # here, before anything moves.
    $dest = Join-Path $versions $ver
    if (Test-Path -LiteralPath $dest) {
      if (Test-DromosInUse $dest) {
        throw "dromos: dromos $ver is running (close it, for example a '$DromosCommandName serve'), then run the installer again"
      }
      $aside = Join-Path $versions ('.replaced-' + (New-DromosId))
      try {
        [IO.Directory]::Move($dest, $aside)
      } catch {
        throw "dromos: dromos $ver is running (close it, for example a '$DromosCommandName serve'), then run the installer again"
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
      throw "dromos: cannot put dromos $ver in $(Get-DromosShortPath $dest)"
    }
    $stage = ''

    # Switch, then tidy. The version this one replaces stays for one rollback, and so does the one before it when this
    # is a re-install of the current version; everything older goes. The junctions are absolute, as install.sh's links are.
    $was = ''
    $wasTarget = Get-DromosLinkTarget $currentLink
    if ($wasTarget) { $was = Split-Path -Leaf $wasTarget }
    $previous = $was
    if ($was -eq $ver) {
      $oldPrevious = Get-DromosLinkTarget $previousLink
      $previous = ''
      if ($oldPrevious) { $previous = Split-Path -Leaf $oldPrevious }
    }
    Remove-DromosLink $currentLink
    New-Item -ItemType Junction -Path $currentLink -Target $dest | Out-Null

    $inUse = @()
    foreach ($dir in @(Get-ChildItem -LiteralPath $versions -Directory -Force)) {
      if ($dir.Name.StartsWith('.')) { continue }
      if ($dir.Name -eq $ver -or $dir.Name -eq $previous) { continue }
      if (Test-DromosInUse $dir.FullName) { $inUse += $dir.Name; continue }
      if (-not (Remove-DromosVersion $dir.FullName $versions)) { $inUse += $dir.Name }
    }
    if ($inUse.Count -gt 0) {
      Show-DromosLine "kept $($inUse -join ', ') for now, because a running dromos is using it. The next install removes it"
    }
    # Set-aside and half-unpacked folders from earlier runs (a dromos was still running from one) go once it has stopped,
    # and so do the ones `dromos update` set aside (.retired-*).
    $leftovers = Get-ChildItem -LiteralPath $versions -Directory -Force |
      Where-Object { $_.Name -like '.replaced-*' -or $_.Name -like '.incoming-*' -or $_.Name -like '.retired-*' }
    foreach ($leftover in @($leftovers)) {
      if (-not (Test-DromosInUse $leftover.FullName)) {
        Remove-Item -LiteralPath $leftover.FullName -Recurse -Force -ErrorAction SilentlyContinue
      }
    }

    Remove-DromosLink $previousLink
    if ($previous -and $previous -ne $ver -and (Test-Path -LiteralPath (Join-Path $versions $previous))) {
      New-Item -ItemType Junction -Path $previousLink -Target (Join-Path $versions $previous) | Out-Null
    }

    $shown = Get-DromosShortPath $dest
    if ($was -and $was -ne $ver) {
      Show-DromosLine "installed dromos $ver in $shown (was $was)"
    } else {
      Show-DromosLine "installed dromos $ver in $shown"
    }
    Add-DromosPath
    Show-DromosLine "done. Try: $DromosCommandName doctor"
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
function Install-Dromos {
  try {
    $null = Invoke-DromosInstall
    return $true
  } catch {
    $message = [string]$_.Exception.Message
    if ($message -notlike 'dromos:*') { $message = "dromos: $message" }
    Write-Host $message -ForegroundColor Red
    return $false
  }
}

$DromosRanAsFile = [bool](Get-Command Install-Dromos -CommandType Function).ScriptBlock.File
if (-not (Install-Dromos) -and $DromosRanAsFile) { exit 1 }
