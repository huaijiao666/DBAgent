<#
.SYNOPSIS
    Run DBAgent with only the current process's temporary provider environment.

.DESCRIPTION
    This wrapper invokes the one-shot `dbagent` command, which reads provider
    configuration only from the current process environment. Use DBA for the
    repository-local ignored config.toml workflow. The script never writes
    credentials.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Task,

    [Parameter(ValueFromRemainingArguments = $true, Position = 1)]
    [string[]]$DBAgentArguments = @(),

    [string]$DBAgentExecutable = (Join-Path $PSScriptRoot '..\.venv\Scripts\dbagent.exe')
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if (-not (Test-Path -LiteralPath $DBAgentExecutable -PathType Leaf)) {
    throw "DBAgent executable was not found: $DBAgentExecutable"
}

& $DBAgentExecutable $Task @DBAgentArguments
exit $LASTEXITCODE
