# Run the three extra Task 2 dev passes back to back on the one GPU, then score
# each against Qwen3.8. One model fits on the 3090 at a time, so each server is
# started, used and stopped before the next:
#
#   1. LLaVA-Video-7B        vLLM 0.12 in WSL2, port 8009   (~22 min)
#   2. InternVL3.5-30B-A3B   llama.cpp b10976, port 8013    (~22 min)
#   3. VideoLLaMA3-7B        in-process, .venv-videollama3  (~55 min)
#
# Resumable: every pass skips queries already in its cache, so re-running after
# an interruption picks up where it stopped. Usage, from any directory:
#
#   powershell -ExecutionPolicy Bypass -File E:\TREC_2026\VQA\run.dev.extra3.ps1
#
# Pause SaladCloud first -- the script refuses to start a model on a busy card.

$ErrorActionPreference = "Continue"
$root = "E:\TREC_2026"
Set-Location $root

$common = @("--queries", "eval.2025.mc\sample400.seed2026.queries.json",
            "--frames", "videos_GT_2025\frames",
            "--transcripts", "videos_GT_2025\transcription_openwhisper",
            "--ignore-routing")

function Wait-Url([string]$url, [int]$minutes) {
    $deadline = (Get-Date).AddMinutes($minutes)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest $url -UseBasicParsing -TimeoutSec 5 | Out-Null
            return
        } catch {
            Start-Sleep 5
        }
    }
    throw "no answer from $url after $minutes min"
}

function Wait-GpuFree([int]$seconds = 120) {
    $deadline = (Get-Date).AddSeconds($seconds)
    $used = 0
    while ((Get-Date) -lt $deadline) {
        $used = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
        if ($used -lt 3000) { return }
        Start-Sleep 5
    }
    throw ("GPU still has $used MiB in use -- pause SaladCloud or stop the " +
           "leftover server, then re-run (finished queries are cached)")
}

function Step([string]$text) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "$(Get-Date -Format 'HH:mm:ss')  $text"
    Write-Host ("=" * 70)
}

# 1. LLaVA-Video --------------------------------------------------------------
Step "1/3 LLaVA-Video-7B: starting vLLM in WSL2"
# The cosmos-reason2 service starts with the distro and takes ~15 GB; stop it
# before checking the card, not after.
wsl.exe -d Ubuntu-24.04 -- systemctl --user stop cosmos-reason2
Wait-GpuFree
$llava = Start-Process wsl.exe -PassThru -WindowStyle Hidden -ArgumentList @(
    "-d", "Ubuntu-24.04", "--", "bash", "/mnt/e/TREC_2026/VQA/serve.llavavideo.sh")
Wait-Url "http://localhost:8009/v1/models" 6
wsl.exe -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/dev.llavavideo.sh
wsl.exe -d Ubuntu-24.04 -- pkill -f vllm
Stop-Process -Id $llava.Id -Force -ErrorAction SilentlyContinue

# 2. InternVL3.5-30B-A3B -------------------------------------------------------
Step "2/3 InternVL3.5-30B-A3B: starting llama-server"
Wait-GpuFree
$gguf = "$root\models\InternVL3_5-30B-A3B-GGUF"
$internvl = Start-Process "$root\tools\llama-b10976\llama-server.exe" -PassThru `
    -WindowStyle Hidden -ArgumentList @(
    "-m", "$gguf\OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf",
    "--mmproj", "$gguf\mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf",
    "--jinja", "-ngl", "all", "-fa", "on", "-c", "32768", "-np", "2",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--alias", "internvl3.5-30b-a3b", "--port", "8013", "--no-webui",
    "--log-file", "$root\logs\internvl.server.log")
Wait-Url "http://localhost:8013/health" 6
# --frame-size 448: one InternVL tile per frame, as its own video path does;
# llama.cpp otherwise tiles each 504x896 frame five times (~1,280 tokens).
& "$root\.venv\Scripts\python.exe" -u VQA\rank.options.mc.muse.py `
    --model internvl3.5-30b-a3b --host http://localhost:8013 `
    --frame-size 448 --temperature 0.2 --num-predict 256 --workers 2 `
    @common --out eval.2025.mc\internvl.dev.run.json `
    --cache eval.2025.mc\internvl.dev.cache.jsonl 2>&1 |
    Tee-Object -FilePath logs\dev.mc.internvl.log -Append
Stop-Process -Id $internvl.Id -Force -ErrorAction SilentlyContinue

# 3. VideoLLaMA3-7B ------------------------------------------------------------
Step "3/3 VideoLLaMA3-7B: in-process"
Wait-GpuFree
& "$root\.venv-videollama3\Scripts\python.exe" -u VQA\rank.options.mc.videollama3.py `
    @common --out eval.2025.mc\videollama3.dev.run.json `
    --cache eval.2025.mc\videollama3.dev.cache.jsonl 2>&1 |
    Tee-Object -FilePath logs\dev.mc.videollama3.log -Append

# Score ------------------------------------------------------------------------
Step "Scores against Qwen3.8 (0.800 Top-1 / 0.884 MRR)"
foreach ($name in "llavavideo", "internvl", "videollama3") {
    & "$root\.venv\Scripts\python.exe" VQA\eval.mc.dev.py score `
        --run "eval.2025.mc\$name.dev.run.json" `
        --baseline eval.2025.mc\qwen38.dev.run.json 2>&1 |
        Tee-Object -FilePath logs\dev.mc.extra3.scores.log -Append
}
