param(
    [int]$MaxParallel = 2,
    [int]$ThreadsPerEstimator = 2,
    [int]$VotingJobs = 2
)

$ErrorActionPreference = 'Stop'
$scriptPath = Join-Path $PSScriptRoot 'run_primarykey_history_factorial.py'
$workspace = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$logDir = Join-Path $PSScriptRoot 'tmp\primarykey_history_factorial\logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

$tasks = @()
foreach ($condition in @(
    'group_no_history',
    'stratified_no_history',
    'group_prior_self_history'
)) {
    foreach ($fold in 1..5) {
        $tasks += [pscustomobject]@{ Condition = $condition; Fold = $fold }
    }
}

$pending = [System.Collections.Generic.Queue[object]]::new()
foreach ($task in $tasks) {
    $pending.Enqueue($task)
}
$running = @()
$failed = @()
$completed = 0
$startedAt = Get-Date

while ($pending.Count -gt 0 -or $running.Count -gt 0) {
    while ($pending.Count -gt 0 -and $running.Count -lt $MaxParallel) {
        $task = $pending.Dequeue()
        $stem = "$($task.Condition)_outer$($task.Fold)"
        $stdout = Join-Path $logDir "$stem.out.log"
        $stderr = Join-Path $logDir "$stem.err.log"
        $quotedScriptPath = '"' + $scriptPath + '"'
        $arguments = @(
            $quotedScriptPath,
            '--condition', $task.Condition,
            '--outer-fold', $task.Fold,
            '--threads-per-estimator', $ThreadsPerEstimator,
            '--voting-jobs', $VotingJobs
        )
        $process = Start-Process -FilePath 'python' -ArgumentList $arguments -WorkingDirectory $workspace -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        $resultDir = Join-Path $PSScriptRoot 'tmp\primarykey_history_factorial'
        $running += [pscustomobject]@{
            Task = $task
            Process = $process
            Started = Get-Date
            Stdout = $stdout
            Stderr = $stderr
            ExpectedMetric = (Join-Path $resultDir "$stem`_metrics.csv")
            ExpectedPrediction = (Join-Path $resultDir "$stem`_predictions.csv.gz")
            ExpectedManifest = (Join-Path $resultDir "$stem`_manifest.json")
        }
        Write-Output "START $stem pid=$($process.Id)"
    }

    Start-Sleep -Seconds 5
    $stillRunning = @()
    foreach ($item in $running) {
        if ($item.Process.HasExited) {
            $item.Process.Refresh()
            $stem = "$($item.Task.Condition)_outer$($item.Task.Fold)"
            $elapsed = [math]::Round(((Get-Date) - $item.Started).TotalSeconds, 1)
            $stderrLength = if (Test-Path $item.Stderr) { (Get-Item $item.Stderr).Length } else { 0 }
            $outputsComplete = (
                (Test-Path $item.ExpectedMetric) -and
                (Test-Path $item.ExpectedPrediction) -and
                (Test-Path $item.ExpectedManifest)
            )
            if ($outputsComplete -and $stderrLength -eq 0) {
                $completed += 1
                Write-Output "DONE $stem seconds=$elapsed completed=$completed/$($tasks.Count)"
            }
            else {
                $failed += $item
                Write-Output "FAILED $stem exit=$($item.Process.ExitCode) stderr=$($item.Stderr)"
            }
        }
        else {
            $stillRunning += $item
        }
    }
    $running = $stillRunning
}

if ($failed.Count -gt 0) {
    throw "$($failed.Count) factorial experiment tasks failed"
}

$totalMinutes = [math]::Round(((Get-Date) - $startedAt).TotalMinutes, 2)
Write-Output "QUEUE COMPLETE tasks=$completed minutes=$totalMinutes"
