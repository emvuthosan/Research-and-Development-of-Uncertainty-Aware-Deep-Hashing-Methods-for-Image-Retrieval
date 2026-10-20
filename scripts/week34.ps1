# Week 3-4 runs on Windows (PowerShell). Run from the repo root:
#   .\scripts\week34.ps1 smoke          # 10-epoch check on the laptop GPU
#   .\scripts\week34.ps1 A              # A, B, C, D, eval100, report (see scripts/week34.sh)
#   $env:DATASET="nuswide_21"; .\scripts\week34.ps1 A
param([Parameter(Mandatory=$true)][string[]]$Jobs)
$ErrorActionPreference = "Stop"

$Dataset = if ($env:DATASET) { $env:DATASET } else { "imagenet" }
$Bit     = if ($env:BIT)     { $env:BIT -split " " } else { @("16") }
$Epoch   = if ($env:EPOCH)   { $env:EPOCH } else { "100" }
$Seed    = if ($env:SEED)    { $env:SEED } else { "0" }
$Workers = if ($env:WORKERS) { $env:WORKERS } else { "2" }
$Common  = @("--dataset", $Dataset, "--bit") + $Bit + @("--epoch", $Epoch, "--seed", $Seed, "--num_workers", $Workers,
             "--val", "--no-pairwise", "--no-tqdm", "--rt_st", "ham", "--sample", "1")

function Run([string[]]$cmd) {
    Write-Host ">>> python $($cmd -join ' ')"
    & python @cmd
    if ($LASTEXITCODE -ne 0) { throw "failed: $($cmd -join ' ')" }
}

foreach ($job in $Jobs) {
    switch ($job) {
        "smoke" { Run (@("CSQ_HashUQ.py", "--grad_est", "cf", "--KL_regularization", "1", "--dataset", $Dataset, "--bit", "16",
                         "--epoch", "10", "--seed", "99", "--num_workers", $Workers, "--val", "--no-pairwise", "--no-tqdm",
                         "--rt_st", "ham", "--sample", "1")) }
        "A" { Run (@("CSQ_HashUQ.py", "--grad_est", "cf", "--KL_regularization", "1") + $Common) }
        "B" { Run (@("CSQ_HashUQ.py", "--grad_est", "st", "--KL_regularization", "1") + $Common) }
        "C" { Run (@("CSQ_HashUQ.py", "--grad_est", "st", "--KL_regularization", "0") + $Common) }
        "D" { Run (@("GreedyHash_HashUQ.py", "--grad_est", "st", "--KL_regularization", "1") + $Common) }
        "eval100" {
            $E = @("--dataset", $Dataset, "--bit") + $Bit + @("--epoch", $Epoch, "--seed", $Seed, "--num_workers", $Workers,
                   "--val", "--no-pairwise", "--no-tqdm", "--rt_st", "hamuct", "--sample", "100", "--sample_method", "MCD", "--no-train")
            Run (@("CSQ_HashUQ.py", "--grad_est", "cf", "--KL_regularization", "1") + $E)
            Run (@("GreedyHash_HashUQ.py", "--grad_est", "st", "--KL_regularization", "1") + $E)
        }
        "report" {
            New-Item -ItemType Directory -Force results | Out-Null
            Run @("summarize_results.py", "saved_model", "--out", "results/summary")
            Run @("plot_curves.py", "saved_model", "--filter", "dataset_$Dataset", "--out", "results/curves_$Dataset.png")
        }
        default { throw "Unknown job '$job' (smoke, A, B, C, D, eval100, report)" }
    }
}
