<#
.SYNOPSIS
    Run Forge with a temporary provider environment.

.DESCRIPTION
    Reads only the provider URL and bearer token from the supplied TOML file,
    exposes them to the Forge child process through environment variables, and
    restores the caller's process environment before exiting. The token is never
    written to the repository or printed.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Task,

    [Parameter(ValueFromRemainingArguments = $true, Position = 1)]
    [string[]]$ForgeArguments = @(),

    [string]$ConfigPath = 'C:\Users\李怀椒\Downloads\WeChat Files\wxid_8lhimj8hmlcv22\FileStorage\File\2026-03\sxdt\config.toml',

    [ValidateSet('gpt-5.6-luna')]
    [string]$Model = 'gpt-5.6-luna',

    [ValidateSet('none', 'low', 'medium', 'high', 'xhigh', 'max')]
    [string]$ReasoningEffort = 'max',

    [string]$ForgeExecutable = (Join-Path $PSScriptRoot '..\.venv\Scripts\forge.exe')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-TomlStringValue {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    $line = Get-Content -LiteralPath $ConfigPath |
        Where-Object { $_ -match "^\s*$Name\s*=" } |
        Select-Object -First 1
    if ($null -eq $line) {
        throw "Required provider setting '$Name' is missing from the config file."
    }

    $value = ($line -split '=', 2)[1].Trim()
    if ($value.Length -ge 2 -and
        (($value.StartsWith('"') -and $value.EndsWith('"')) -or
         ($value.StartsWith("'") -and $value.EndsWith("'")))) {
        $value = $value.Substring(1, $value.Length - 2)
    }
    if ([string]::IsNullOrWhiteSpace($value)) {
        throw "Required provider setting '$Name' is empty."
    }
    return $value
}

if (-not (Test-Path -LiteralPath $ConfigPath -PathType Leaf)) {
    throw "Provider config file was not found: $ConfigPath"
}
if (-not (Test-Path -LiteralPath $ForgeExecutable -PathType Leaf)) {
    throw "Forge executable was not found: $ForgeExecutable"
}

$baseUrl = Get-TomlStringValue -Name 'base_url'
$token = Get-TomlStringValue -Name 'experimental_bearer_token'
if ($token -eq '<redacted>') {
    throw 'Provider bearer token is not usable.'
}

$environmentNames = @(
    'OPENAI_API_KEY',
    'FORGE_BASE_URL',
    'FORGE_API_MODE',
    'FORGE_MODEL',
    'FORGE_REASONING_EFFORT'
)
$previousEnvironment = @{}
foreach ($name in $environmentNames) {
    $previousEnvironment[$name] = [Environment]::GetEnvironmentVariable(
        $name,
        'Process'
    )
}

$exitCode = 1
try {
    [Environment]::SetEnvironmentVariable('OPENAI_API_KEY', $token, 'Process')
    [Environment]::SetEnvironmentVariable('FORGE_BASE_URL', $baseUrl, 'Process')
    [Environment]::SetEnvironmentVariable(
        'FORGE_API_MODE',
        'chat_completions',
        'Process'
    )
    [Environment]::SetEnvironmentVariable('FORGE_MODEL', $Model, 'Process')
    [Environment]::SetEnvironmentVariable(
        'FORGE_REASONING_EFFORT',
        $ReasoningEffort,
        'Process'
    )

    & $ForgeExecutable $Task @ForgeArguments
    $exitCode = $LASTEXITCODE
}
finally {
    foreach ($name in $environmentNames) {
        $previousValue = $previousEnvironment[$name]
        if ([string]::IsNullOrWhiteSpace($previousValue)) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        }
        else {
            [Environment]::SetEnvironmentVariable($name, $previousValue, 'Process')
        }
    }
    Remove-Variable token -ErrorAction SilentlyContinue
}

exit $exitCode
