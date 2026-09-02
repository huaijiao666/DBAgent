<#
.SYNOPSIS
    Install this checkout and make the DBA command available in new terminals.

.DESCRIPTION
    The project remains isolated in its repository virtual environment. This
    script optionally creates that environment, installs the package in
    editable mode, and adds only its Scripts directory to the current user's
 PATH. API credentials are not read or written by this script; DBA loads only
 the Git-ignored provider files at this repository root when it starts.
#>

[CmdletBinding()]
param(
    [switch]$Install
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$venvRoot = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvRoot 'Scripts\python.exe'
$venvScripts = Join-Path $venvRoot 'Scripts'
$dbaExecutable = Join-Path $venvScripts 'dba.exe'

# Windows cannot replace a running console launcher. Failing before pip starts
# keeps the environment intact and explains the only required recovery action.
$runningDba = @(
    Get-Process -Name 'dba' -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Path -and [string]::Equals(
                $_.Path,
                $dbaExecutable,
                [StringComparison]::OrdinalIgnoreCase
            )
        }
)
if ($runningDba.Count -gt 0) {
    throw 'DBA is currently running from this virtual environment. Type /exit in that DBA session, then run this installer again.'
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    if (-not $Install) {
        throw "Python environment is missing. Run .\scripts\install-dba.ps1 -Install once."
    }

    $pythonLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($null -eq $pythonLauncher) {
        throw "The Windows Python launcher 'py' was not found. Install Python 3.11+ first."
    }
    & $pythonLauncher.Source '-3.11' '-m' 'venv' $venvRoot
    if ($LASTEXITCODE -ne 0) {
        throw "Creating the project virtual environment failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    throw "The project virtual environment was not created at $venvPython."
}

$installTarget = "$repoRoot[dev]"
& $venvPython '-m' 'pip' 'install' '--disable-pip-version-check' '-e' $installTarget
if ($LASTEXITCODE -ne 0) {
    throw "Installing the editable DBA package failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path -LiteralPath $dbaExecutable -PathType Leaf)) {
    throw "The DBA launcher was not created at $dbaExecutable."
}

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$pathEntries = @()
if (-not [string]::IsNullOrWhiteSpace($userPath)) {
    $pathEntries = @(
        $userPath -split ';' |
            Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
            ForEach-Object { $_.Trim() }
    )
}

$normalizedScripts = $venvScripts.TrimEnd('\')
$alreadyRegistered = $pathEntries |
    Where-Object {
        [string]::Equals(
            $_.TrimEnd('\'),
            $normalizedScripts,
            [StringComparison]::OrdinalIgnoreCase
        )
    }
if ($null -eq $alreadyRegistered -or @($alreadyRegistered).Count -eq 0) {
    $pathEntries += $venvScripts
    [Environment]::SetEnvironmentVariable(
        'Path',
        ($pathEntries -join ';'),
        'User'
    )
}

# Make the command usable immediately in this PowerShell process as well.
$processPathEntries = @($env:Path -split ';' | Where-Object { $_ })
$processHasScripts = $processPathEntries |
    Where-Object {
        [string]::Equals(
            $_.TrimEnd('\'),
            $normalizedScripts,
            [StringComparison]::OrdinalIgnoreCase
        )
    }
if ($null -eq $processHasScripts -or @($processHasScripts).Count -eq 0) {
    $env:Path = "$venvScripts;$env:Path"
}

Write-Output "DBA is available from new terminals through: $venvScripts"
Write-Output "Example: cd C:\any\workspace; DBA --workspace ."
Write-Output 'DBA reads an optional ignored config.toml and api_key.txt at this repository root; no API key was written.'
