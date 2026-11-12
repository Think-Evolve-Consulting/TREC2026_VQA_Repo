# Finish the extra 2026 Task 2 passes on the one GPU, unattended, and validate
# each run file. One model fits on the 3090 at a time, so in order:
#
#   0. InternVL3.5-30B-A3B  wait for the running pass; resume it if it stopped
#                           short; stop its llama-server; validate
#   1. LLaVA-Video-7B       vLLM 0.12 in WSL2, port 8009           (~2.7 h)
#   2. VideoLLaMA3-7B       in-process, .venv-videollama3          (~6-7 h)
#
# Every pass resumes from its cache and a finished one is skipped, so this can
# be re-run safely after any interruption. Launch detached, so it outlives the
# shell that started it:
#
#   Start-Process powershell -WindowStyle Hidden -ArgumentList `
#       "-ExecutionPolicy Bypass -File E:\TREC_2026\VQA\run2026.extra2.ps1"
#
# Its own log: logs\run2026.extra2.log. Per-model logs:
# logs\rank.options.mc.task2.{internvl,llavavideo,videollama3}.log.

$ErrorActionPreference = "Continue"
$root = "E:\TREC_2026"
Set-Location $root
Start-Transcript -Path "$root\logs\run2026.extra2.log" -Append | Out-Null

$py = "$root\.venv\Scripts\python.exe"
$queries = "vqa.2026.testingDataset.Task2_v2.json"
$total = 2443

function Step([string]$text) {
    Write-Host ""
    Write-Host ("=" * 70)
    Write-Host "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')  $text"
    Write-Host ("=" * 70)
}

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

function Wait-GpuFree([int]$seconds = 180) {
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

function Cached([string]$path) {
    if (-not (Test-Path $path)) { return 0 }
    [int](& $py -c "import json,sys; print(len({json.loads(l)['Q_ID'] for l in open(sys.argv[1], encoding='utf-8') if l.strip()}))" $path)
}

function Validate([string]$run) {
    $report = $run -replace '\.json$', '.validation.json'
    & $py validate_vqa_mc_2026.py --queries $queries --run $run `
        --warnings-as-errors --report-json $report 2>&1 | Select-Object -Last 12 | Out-Host
    if ($LASTEXITCODE -eq 0) {
        Write-Host "VALIDATION: VALID -- $run"
    } else {
        Write-Host "VALIDATION: INVALID -- $run (exit $LASTEXITCODE)"
    }
}

function Start-InternVL {
    $gguf = "$root\models\InternVL3_5-30B-A3B-GGUF"
    Start-Process "$root\tools\llama-b10976\llama-server.exe" -WindowStyle Hidden -ArgumentList @(
        "-m", "$gguf\OpenGVLab_InternVL3_5-30B-A3B-Q4_K_M.gguf",
        "--mmproj", "$gguf\mmproj-OpenGVLab_InternVL3_5-30B-A3B-f16.gguf",
        "--jinja", "-ngl", "all", "-fa", "on", "-c", "32768", "-np", "2",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--alias", "internvl3.5-30b-a3b", "--port", "8013", "--no-webui",
        "--log-file", "$root\logs\internvl.server.2026.log") | Out-Null
    Wait-Url "http://localhost:8013/health" 6
}

# 0. InternVL3.5 -----------------------------------------------------------------
Step "0/2 InternVL3.5-30B-A3B: waiting for the running pass to finish"
while (Get-CimInstance Win32_Process -Filter "Name like 'python%'" |
       Where-Object { $_.CommandLine -match 'rank\.options\.mc\.muse\.py' }) {
    Start-Sleep 30
}

$done = Cached "rankings.task2.internvl.cache.jsonl"
if ($done -lt $total) {
    Step "InternVL stopped at $done/$total -- resuming from the cache"
    try {
        Invoke-RestMethod http://localhost:8013/health -TimeoutSec 5 | Out-Null
    } catch {
        Wait-GpuFree
        Start-InternVL
    }
    & $py -u VQA\rank.options.mc.muse.py --model internvl3.5-30b-a3b `
        --host http://localhost:8013 --frame-size 448 --temperature 0.2 `
        --num-predict 256 --workers 2 --out internvl.run.task2.json `
        --cache rankings.task2.internvl.cache.jsonl 2>&1 |
        Tee-Object -FilePath logs\rank.options.mc.task2.internvl.log -Append
}
Get-Process llama-server -ErrorAction SilentlyContinue | Stop-Process -Force
Write-Host "InternVL cached: $(Cached 'rankings.task2.internvl.cache.jsonl')/$total"
Validate "internvl.run.task2.json"

# 1. LLaVA-Video -----------------------------------------------------------------
Step "1/2 LLaVA-Video-7B: vLLM in WSL2"
if ((Cached "rankings.task2.llavavideo.cache.jsonl") -lt $total) {
    # The cosmos-reason2 service starts with the distro and takes ~15 GB.
    wsl.exe -d Ubuntu-24.04 -- systemctl --user stop cosmos-reason2
    Wait-GpuFree
    $llava = Start-Process wsl.exe -PassThru -WindowStyle Hidden -ArgumentList @(
        "-d", "Ubuntu-24.04", "--", "bash", "/mnt/e/TREC_2026/VQA/serve.llavavideo.sh")
    Wait-Url "http://localhost:8009/v1/models" 8
    wsl.exe -d Ubuntu-24.04 -- bash /mnt/e/TREC_2026/VQA/run2026.llavavideo.sh
    wsl.exe -d Ubuntu-24.04 -- pkill -f vllm
    Stop-Process -Id $llava.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep 10
}
Write-Host "LLaVA-Video cached: $(Cached 'rankings.task2.llavavideo.cache.jsonl')/$total"
Validate "llavavideo.run.task2.json"

# 2. VideoLLaMA3 -----------------------------------------------------------------
Step "2/2 VideoLLaMA3-7B: in-process"
if ((Cached "rankings.task2.videollama3.cache.jsonl") -lt $total) {
    Wait-GpuFree
    & "$root\.venv-videollama3\Scripts\python.exe" -u VQA\rank.options.mc.videollama3.py `
        --out videollama3.run.task2.json `
        --cache rankings.task2.videollama3.cache.jsonl 2>&1 |
        Tee-Object -FilePath logs\rank.options.mc.task2.videollama3.log -Append
}
Write-Host "VideoLLaMA3 cached: $(Cached 'rankings.task2.videollama3.cache.jsonl')/$total"
Validate "videollama3.run.task2.json"

Step "ALL DONE"
Stop-Transcript | Out-Null
