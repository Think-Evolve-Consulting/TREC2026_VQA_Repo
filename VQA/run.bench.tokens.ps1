# Measure prefill and decode tokens/s for every VLM in this project, one model
# on the card at a time. Each server is started, benched and stopped before the
# next one loads:
#
#   1. Cosmos-Reason2-2B        vLLM 0.12 in WSL2, port 8008
#   2. Qwen3-VL-8B-Instruct     vLLM 0.12 in WSL2, port 8009
#   3. LLaVA-Video-7B           vLLM 0.12 in WSL2, port 8009
#   4. Gemma 4 12B QAT INT4     vLLM 0.27 in WSL2, port 8010
#   5. Qwen3.8-27B INT4         vLLM 0.27 in WSL2, port 8011
#   6. Muse Glimmer 30B         llama.cpp b10976, port 8012
#   7. InternVL3.5-30B-A3B      llama.cpp b10976, port 8013
#   8. VideoLLaMA3-7B           in-process, .venv-videollama3
#
# Usage, from any directory:
#
#   powershell -ExecutionPolicy Bypass -File E:\TREC_2026\VQA\run.bench.tokens.ps1
#   ... -Task 1                      # answer generation instead of MC ranking
#   ... -Models cosmos,qwen38        # just those two
#   ... -Concurrency 4               # also report aggregate throughput
#   ... -Spread corpus               # the corpus's own duration mix
#   ... -ForceLength 256             # pin every completion, for exact comparison
#
# Each model is measured on the same ten TREC 2026 queries -- the same clips
# for either task, since the two share Q_IDs -- chosen to span the clip
# duration range (1 s to 389 s). Cost per model is a model load (1-6 min)
# plus ten real requests, so the full sweep is roughly an hour -- longer if
# Cosmos and Muse spend their reasoning budgets on the long clips.
#
# Pause SaladCloud first -- the script refuses to start a model on a busy card.
# Results append to E:\TREC_2026\bench.tokens.jsonl; the table is also printed.

param(
    [string[]] $Models = @("cosmos", "qwen3vl", "llavavideo", "gemma4",
                           "qwen38", "muse", "internvl", "videollama3"),
    [ValidateSet(1, 2)] [int] $Task = 2,
    [int] $Videos = 10,
    [ValidateSet("duration", "corpus")] [string] $Spread = "duration",
    [int] $Frames = 16,
    [int] $Repeats = 1,
    [int] $Concurrency = 1,
    [int] $ForceLength = 0,
    [switch] $Sweep,
    [int] $LoadMinutes = 8
)

$ErrorActionPreference = "Continue"
$root = "E:\TREC_2026"
Set-Location $root
New-Item -ItemType Directory -Force "$root\logs" | Out-Null

# Every model sees the same clips: the selection is deterministic given
# --videos and --spread, so the numbers compare row to row.
$flags = @("--task", $Task, "--videos", $Videos, "--spread", $Spread,
           "--frames", $Frames, "--repeats", $Repeats,
           "--concurrency", $Concurrency)
if ($Sweep) { $flags += "--sweep" }
if ($ForceLength -gt 0) { $flags += @("--force-length", $ForceLength) }

function Step([string]$text) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  $text"
    Write-Host ("=" * 70)
}

function Wait-Url([string]$url, [int]$minutes) {
    $deadline = (Get-Date).AddMinutes($minutes)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 5 | Out-Null
            return $true
        } catch {
            Start-Sleep 5
        }
    }
    Write-Host "  no answer from $url after $minutes min -- skipping"
    return $false
}

function Wait-GpuFree([int]$seconds = 180) {
    $deadline = (Get-Date).AddSeconds($seconds)
    $used = 0
    while ((Get-Date) -lt $deadline) {
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($used -lt 3000) { return $true }
        Start-Sleep 5
    }
    Write-Host "  GPU still has $used MiB in use -- skipping this model"
    return $false
}

# A vLLM model: start the server inside WSL2, bench it from inside WSL2 (the
# server opens each clip off the /mnt/e path in the request), then stop it.
function Bench-Vllm([string]$name, [int]$port) {
    Step "$name : starting vLLM in WSL2 (port $port)"
    wsl.exe -d Ubuntu-24.04 -- systemctl --user stop cosmos-reason2
    if (-not (Wait-GpuFree)) { return }

    $server = Start-Process wsl.exe -PassThru -WindowStyle Hidden -ArgumentList @(
        "-d", "Ubuntu-24.04", "--", "bash",
        "/mnt/e/TREC_2026/VQA/serve.vllm.sh", $name)

    if (Wait-Url "http://127.0.0.1:$port/v1/models" $LoadMinutes) {
        wsl.exe -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/bench.tokens.sh `
            $name @flags
    }

    wsl.exe -d Ubuntu-24.04 -- pkill -f vllm
    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 5
}

# A llama.cpp model: llama-server runs on the Windows side, so the bench runs
# in the project venv.
function Bench-LlamaCpp([string]$name, [int]$port, [string[]]$serverArgs) {
    Step "$name : starting llama-server (port $port)"
    if (-not (Wait-GpuFree)) { return }

    $server = Start-Process "$root\tools\llama-b10976\llama-server.exe" -PassThru `
        -WindowStyle Hidden -ArgumentList $serverArgs

    # No Tee-Object: PowerShell 5.1 writes it as UTF-16LE, which grep and tail
    # cannot read. The durable record is bench.tokens.jsonl.
    if (Wait-Url "http://127.0.0.1:$port/health" $LoadMinutes) {
        & "$root\.venv\Scripts\python.exe" -u VQA\bench.tokens.py `
            --model $name @flags
    }

    Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 5
}

foreach ($name in $Models) {
    switch ($name) {
        "cosmos"     { Bench-Vllm "cosmos" 8008 }
        "qwen3vl"    { Bench-Vllm "qwen3vl" 8009 }
        "llavavideo" { Bench-Vllm "llavavideo" 8009 }
        "gemma4"     { Bench-Vllm "gemma4" 8010 }
        "qwen38"     { Bench-Vllm "qwen38" 8011 }

        "muse" {
            $gguf = "$root\models\Muse-Glimmer-30B-GGUF"
            Bench-LlamaCpp "muse" 8012 @(
                "-m", "$gguf\Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
                "--mmproj", "$gguf\mmproj-Muse-Glimmer-30B-BF16.gguf",
                "--jinja", "--reasoning-format", "auto", "-ngl", "all",
                "-fa", "on", "-c", "49152", "-np", "3",
                "--image-max-tokens", "256",
                "--alias", "muse-glimmer-30b", "--port", "8012", "--no-webui",
                "--log-file", "$root\logs\bench.server.muse.log")
        }

        "internvl" {
            # --frame-size 448 is set in the bench registry: llama.cpp tiles
            # every InternVL image into 448 px crops and --image-max-tokens
            # does not cap it, so a 504x896 frame costs ~1,280 tokens and
            # 16 frames overflow the window.
            $gguf = "$root\models\InternVL3_5-30B-A3B-GGUF"
            Bench-LlamaCpp "internvl" 8013 @(
                "-m", "$gguf\OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf",
                "--mmproj", "$gguf\mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf",
                "--jinja", "-ngl", "all", "-fa", "on", "-c", "32768", "-np", "2",
                "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                "--alias", "internvl3.5-30b-a3b", "--port", "8013", "--no-webui",
                "--log-file", "$root\logs\bench.server.internvl.log")
        }

        "videollama3" {
            Step "videollama3 : in-process, .venv-videollama3"
            if (Wait-GpuFree) {
                & "$root\.venv-videollama3\Scripts\python.exe" -u VQA\bench.tokens.py `
                    --model videollama3 @flags
            }
        }

        default { Write-Host "unknown model '$name' -- skipping" }
    }
}

Step "done"
Write-Host "Results: $root\bench.tokens.jsonl"
Write-Host "Summary table: E:\TREC_2026\.venv\Scripts\python.exe VQA\bench.tokens.py --report"
