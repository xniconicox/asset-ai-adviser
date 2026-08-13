param(
    [string]$TaskPrefix = "AssetAIAdviser",
    [string]$DailyAt = "19:30",
    [string]$BackupAt = "20:30",
    [string]$WslDistribution = ""
)

$ErrorActionPreference = "Stop"
$PocPath = if ($env:ASSET_POC_PATH) {
    $env:ASSET_POC_PATH
} else {
    (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$DistributionArgument = if ($WslDistribution) { "-d $WslDistribution " } else { "" }

$DailyAction = New-ScheduledTaskAction `
    -Execute "wsl.exe" `
    -Argument "${DistributionArgument}-- bash $PocPath/scripts/run-daily.sh"
$DailyTrigger = New-ScheduledTaskTrigger -Daily -At $DailyAt
$DailySettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 4)
Register-ScheduledTask `
    -TaskName "${TaskPrefix}-Daily" `
    -Action $DailyAction `
    -Trigger $DailyTrigger `
    -Settings $DailySettings `
    -Description "Asset AI Adviser daily data, ranks, DQ, snapshot and PDF report (no LLM usage)" `
    -Force

$BackupAction = New-ScheduledTaskAction `
    -Execute "wsl.exe" `
    -Argument "${DistributionArgument}-- bash $PocPath/scripts/run-backup.sh"
$BackupTrigger = New-ScheduledTaskTrigger `
    -Weekly `
    -WeeksInterval 1 `
    -DaysOfWeek Sunday `
    -At $BackupAt
$BackupSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask `
    -TaskName "${TaskPrefix}-Backup" `
    -Action $BackupAction `
    -Trigger $BackupTrigger `
    -Settings $BackupSettings `
    -Description "Asset AI Adviser weekly DuckDB backup (no LLM usage)" `
    -Force

Write-Host "Registered ${TaskPrefix}-Daily at $DailyAt and weekly backup at $BackupAt."
