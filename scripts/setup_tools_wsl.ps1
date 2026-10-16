# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 edge_vision_pipeline contributors
# Install the native deploy tools (pnnx, ncnn2table, ncnn2int8) inside WSL by running
# scripts/setup_tools_linux.sh there. The Windows pnnx.exe crashes on some machines, so evp uses the WSL build.
param(
    [string]$Distro = "",
    [string]$ToolsHome = ""
)
$ErrorActionPreference = "Stop"

$wsl = Get-Command wsl -ErrorAction SilentlyContinue
if (-not $wsl) { throw "WSL not found. Install it with: wsl --install -d Ubuntu-24.04" }

$repo = Split-Path -Parent $PSScriptRoot
$drive = $repo.Substring(0, 1).ToLower()
$repoWsl = "/mnt/$drive" + ($repo.Substring(2) -replace '\\', '/')

$wslArgs = @()
if ($Distro) { $wslArgs += @("-d", $Distro) }
$envPrefix = @()
if ($ToolsHome) { $envPrefix = @("env", "EVP_TOOLS_HOME=$ToolsHome") }

& wsl @wslArgs --cd $repoWsl --exec @envPrefix bash scripts/setup_tools_linux.sh
if ($LASTEXITCODE -ne 0) { throw "setup_tools_linux.sh failed in WSL (exit $LASTEXITCODE)" }

& wsl @wslArgs --cd $repoWsl --exec bash scripts/build_runtime.sh
if ($LASTEXITCODE -ne 0) { throw "build_runtime.sh failed in WSL (exit $LASTEXITCODE)" }
Write-Host "[OK] tools installed in WSL; check with: evp tools"
