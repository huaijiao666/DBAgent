<#
.SYNOPSIS
    Run Forge with only the current process's temporary provider environment.

.DESCRIPTION
    DBA reads an optional Git-ignored config.toml at the repository root into
    this process only. OPENAI_API_KEY and FORGE_* environment variables remain
    supported as explicit overrides. The script never writes credentials.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Task,

    [Parameter(ValueFromRemainingArguments = $true, Position = 1)]
    [string[]]$ForgeArguments = @(),

    [string]$ForgeExecutable = (Join-Path $PSScriptRoot '..\.venv\Scripts\forge.exe')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $ForgeExecutable -PathType Leaf)) {
    throw "Forge executable was not found: $ForgeExecutable"
}

& $ForgeExecutable $Task @ForgeArguments
exit $LASTEXITCODE
