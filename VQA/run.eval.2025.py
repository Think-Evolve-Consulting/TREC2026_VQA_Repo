#!/usr/bin/env python3
"""Run the whole 2025 ground-truth evaluation, unattended, across two hosts.

Measures the downloaded 2025 GT videos, draws a 150-video sample, and puts all
three answering pipelines over it with full 2026 input parity (Whisper
transcripts, routed inputs, Gemma 4's paraphrase ensemble), then scores every
run against the ground truth.

**Run this from Windows.** The Qwen3-VL and Cosmos clients must run *inside*
WSL2, because they hand the server a ``file://`` URI it has to open itself --
``file:///E:/...`` means nothing to a Linux process -- so the orchestrator
lives on Windows and dispatches those stages through ``wsl.exe``. The three
Gemma 4 stages run from Windows against the WSL2 server on 8010, which is fine
because they send their frames as base64 in the request body and so depend on
nothing but the port. Windows localhost forwards into the distro.

Three model backends contend for one RTX 3090 and none of them can coexist
(each vLLM server takes 16-23 GB). ``ensure_backend()`` is the only thing
allowed to touch them, and the stage order switches the card exactly three
times::

    any  ->  free  -> vllm-gemma4 -> vllm-qwen  ->  vllm-cosmos  ->  free  ->  any
   queries  whisper   route          answers       answers         score    report
   durations          variations     qwen3vl       cosmos
   sample             gemma4
   frames

Gemma 4 moved off Ollama onto vLLM in WSL2 on 15 Aug 2026
(``VQA/GEMMA4_VLLM_SETUP.md``), so the default path no longer touches Ollama
at all. ``--gemma4-backend ollama`` restores it; it serves a different
quantisation (Q4_K_M against Google's QAT INT4), so it is a different run, not
a resumable one.

A stage declaring ``any`` never evicts what is on the card -- the 2026 run's
server may still be serving it, and a CPU-only stage has no business killing
that. ``free`` claims the card without starting a server of its own.

Two hazards this script exists to prevent:

**Q_ID collision.** 2025 Q_IDs are 1..2000 and 2026 Q_IDs are 1..2443, and
every cache in this repo is keyed by bare Q_ID. A 2025 stage pointed at a
default cache would read 2026 answers as finished work and emit a run built
from answers to different questions. Every stage therefore passes every path
flag explicitly, and ``--check-flags`` proves no path flag was left to its
default.

**Silent overwrite.** ``duration.report.py`` defaults to
``--report duration.report.json``, which is the live 2026 artifact
``extract.frames.py`` reads. The guard hashes the 2026 files up front and
re-verifies after *every* stage, so a mistake aborts at the stage that caused
it rather than being discovered at the end.

Usage:
    python VQA/run.eval.2025.py --smoke 5      # rehearse everything, ~30 min
    python VQA/run.eval.2025.py                # the real thing, ~3.5 h
    python VQA/run.eval.2025.py --dry-run      # print the commands, run nothing
    python VQA/run.eval.2025.py --from route   # resume partway
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VQA = ROOT / "VQA"

GT_DIR = ROOT / "videos_GT_2025"
VIDEOS = GT_DIR / "videos"
METADATA = GT_DIR / "metadata"
FRAMES = GT_DIR / "frames"
TRANSCRIPTS = GT_DIR / "transcription_openwhisper"

LOG_DIR = ROOT / "logs" / "eval2025"
STATE_FILE = ROOT / "eval2025.state.json"

WSL_DISTRO = "Ubuntu-24.04"
WSL_PYTHON = "/home/aakash/cosmos-reason2/.venv/bin/python"
WSL_ROOT = "/mnt/e/TREC_2026"
WSL_SERVE_DIR = "/home/aakash/cosmos-reason2"

WINDOWS_PYTHON = ROOT / ".venv" / "Scripts" / "python.exe"
OLLAMA_EXE = Path(os.environ.get("LOCALAPPDATA", r"C:\Users\ADMIN\AppData\Local")) \
    / "Programs" / "Ollama" / "ollama.exe"

QWEN_URL = "http://localhost:8009/v1"
COSMOS_URL = "http://localhost:8008/v1"
GEMMA_URL = "http://localhost:8010/v1"
OLLAMA_URL = "http://localhost:11434"

QWEN_MODEL = "Qwen/Qwen3-VL-8B-Thinking"
COSMOS_MODEL = "nvidia/Cosmos-Reason2-2B"
GEMMA_MODEL = "google/gemma-4-12B-it-qat-w4a16-ct"

# The Ollama tag, kept only for --gemma4-backend ollama. It is a different
# quantisation of the same model (Q4_K_M against Google's QAT INT4), so the two
# are not interchangeable mid-run.
GEMMA_OLLAMA_MODEL = "gemma4:12b"

# Gemma 4 moved off Ollama on 15 Aug 2026 -- VQA/GEMMA4_VLLM_SETUP.md. Its
# venv is separate from the Cosmos/Qwen one (vLLM 0.27.1 against 0.12.0), and
# the two env vars are WSL2 facts rather than Gemma 4 ones: no UVA under
# GPU-PV, and no nvcc for FlashInfer's JIT sampler.
GEMMA_SERVE_DIR = "/home/aakash/gemma4"
GEMMA_SERVE = (
    f"cd {GEMMA_SERVE_DIR} && "
    "VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 "
    f"setsid nohup ./.venv/bin/vllm serve {GEMMA_MODEL} "
    "--host 0.0.0.0 --port 8010 "
    f"--served-model-name {GEMMA_MODEL} "
    f"--allowed-local-media-path {WSL_ROOT} "
    "--max-model-len 16384 --gpu-memory-utilization 0.92 --enforce-eager "
    "--limit-mm-per-prompt '{\"image\": 32}' "
    "--reasoning-parser gemma4 "
    "< /dev/null > /home/aakash/gemma4/eval.log 2>&1 & disown; exit 0"
)

# Copied verbatim from the process that serves the 2026 run. --skip-mm-profiling
# is mandatory: without it vLLM profiles Qwen3-VL's 153,600-token max feature
# size, KV cache computes to -1.20 GiB and the engine refuses to start.
#
# ``setsid`` and ``< /dev/null`` are load-bearing, not decoration. A process
# merely backgrounded from ``wsl.exe -- bash -lc '... &'`` is killed when that
# wsl.exe session ends: the launch returns rc=0, the redirect creates an empty
# log, and nothing is listening 15 minutes later. ``nohup`` alone does not save
# it -- the process must leave the session entirely. (The 2026 server survived
# only because a human started it from an interactive WSL terminal.)
QWEN_SERVE = (
    f"cd {WSL_SERVE_DIR} && "
    f"setsid nohup ./.venv/bin/vllm serve {QWEN_MODEL} "
    "--host 0.0.0.0 --port 8009 "
    f"--served-model-name {QWEN_MODEL} "
    f"--allowed-local-media-path {WSL_ROOT} "
    "--max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager "
    "--media-io-kwargs '{\"video\": {\"num_frames\": -1}}' "
    "--reasoning-parser qwen3 --skip-mm-profiling "
    "< /dev/null > /home/aakash/qwen3vl.eval.log 2>&1 & disown; exit 0"
)

# The systemd user unit is the normal path; this is the fallback if it is not
# installed. Flags match VQA/COSMOS_REASON2_SETUP.md.
COSMOS_SERVE = (
    f"cd {WSL_SERVE_DIR} && "
    f"setsid nohup ./.venv/bin/vllm serve {COSMOS_MODEL} "
    "--host 0.0.0.0 --port 8008 "
    f"--allowed-local-media-path {WSL_ROOT} "
    "--max-model-len 24576 --gpu-memory-utilization 0.75 --enforce-eager "
    "--media-io-kwargs '{\"video\": {\"num_frames\": -1}}' "
    "--reasoning-parser qwen3 "
    "< /dev/null > /home/aakash/cosmos.eval.log 2>&1 & disown; exit 0"
)

# "any" means the stage does not touch the GPU, so whatever is loaded stays
# loaded. That distinction matters: the 2026 run may still be serving, and a
# CPU-only stage must not evict it. "free" means the stage needs the card to
# itself (Whisper, BERTScore) without starting a server of its own.
BACKEND_ANY = "any"
BACKEND_FREE = "free"
BACKEND_OLLAMA = "ollama"
BACKEND_QWEN = "vllm-qwen"
BACKEND_COSMOS = "vllm-cosmos"
BACKEND_GEMMA = "vllm-gemma4"

# The 2026 artifacts that must not change. Hashed before the run and re-checked
# after every stage -- see the module docstring's "silent overwrite".
GUARDED = (
    "duration.report.json",
    "duration.report.html",
    "query.routing.task1.json",
    "query.variations.task1.json",
    "gemma4.run.task1.json",
    "cosmos.run.task1.json",
    "qwen3vl.run.task1.json",
    "answers.task1.cache.jsonl",
    "answers.task1.cosmos.cache.jsonl",
    "answers.task1.qwen3vl.cache.jsonl",
    "vqa.2026.testingDataset.Task1.json",
)

QWEN3VL_2026_CACHE = ROOT / "answers.task1.qwen3vl.cache.jsonl"
QUERIES_2026 = ROOT / "vqa.2026.testingDataset.Task1.json"

SCORER_PACKAGES = (("pandas", "pandas"), ("nltk", "nltk"),
                   ("bert_score", "bert-score"),
                   ("sentence_transformers", "sentence-transformers"))


# ---------------------------------------------------------------------------
# Paths for this run
# ---------------------------------------------------------------------------

@dataclass
class Ctx:
    n: int
    smoke: bool
    seed: int
    dry_run: bool
    force: bool
    workers_qwen: int
    workers_cosmos: int
    workers_gemma: int
    gemma_backend: str
    metrics: str

    def gemma_flags(self) -> list[str]:
        """The backend flags every Gemma 4 stage passes to its script.

        Passed explicitly rather than left to default, for the same reason
        every path flag is: --check-flags proves nothing was inherited, and a
        2025 run must not depend on what the scripts happen to default to.
        """
        if self.gemma_backend == "ollama":
            return ["--backend", "ollama", "--host", OLLAMA_URL,
                    "--model", GEMMA_OLLAMA_MODEL]

        return ["--backend", "vllm", "--host", GEMMA_URL.removesuffix("/v1"),
                "--model", GEMMA_MODEL]

    def backend_for(self, declared: str) -> str:
        """Resolve a stage's declared backend against --gemma4-backend."""
        if declared == BACKEND_GEMMA and self.gemma_backend == "ollama":
            return BACKEND_OLLAMA

        return declared

    @property
    def label(self) -> str:
        return f"smoke{self.n}" if self.smoke else f"sample{self.n}"

    @property
    def tag(self) -> str:
        """Infix that keeps a rehearsal from touching the real caches."""
        return ".smoke" if self.smoke else ""

    # -- shared across smoke and real: media derivatives are a strict subset --
    queries_all = ROOT / "vqa.gt.2025.queries.json"
    pool = ROOT / "vqa.gt.2025.pool.json"
    gt_full = ROOT / "vqa.gt.2025.csv"
    durations = ROOT / "duration.report.gt2025.json"
    durations_html = ROOT / "duration.report.gt2025.html"

    @property
    def queries(self) -> Path:
        return ROOT / f"vqa.2025.{self.label}.json"

    @property
    def gt(self) -> Path:
        return ROOT / f"vqa.gt.2025.{self.label}.csv"

    @property
    def reserves(self) -> Path:
        return ROOT / f"vqa.2025.{self.label}.reserves.json"

    @property
    def sample_report(self) -> Path:
        return ROOT / f"sample.2025.{self.label}.report.json"

    @property
    def sample_report_md(self) -> Path:
        return ROOT / f"sample.2025.{self.label}.report.md"

    @property
    def routing(self) -> Path:
        return ROOT / f"query.routing.2025{self.tag}.json"

    @property
    def routing_cache(self) -> Path:
        return ROOT / f"query.routing.2025{self.tag}.cache.jsonl"

    @property
    def variations(self) -> Path:
        return ROOT / f"query.variations.2025{self.tag}.json"

    @property
    def variations_cache(self) -> Path:
        return ROOT / f"query.variations.2025{self.tag}.cache.jsonl"

    def run_file(self, model: str) -> Path:
        return ROOT / f"{model}.run.2025{self.tag}.json"

    def run_cache(self, model: str) -> Path:
        return ROOT / f"answers.2025.{model}{self.tag}.cache.jsonl"

    def validation(self, model: str) -> Path:
        return ROOT / f"{model}.run.2025{self.tag}.validation.json"

    @property
    def scores(self) -> Path:
        return ROOT / f"eval.2025{self.tag}.scores.json"

    @property
    def scores_csv(self) -> Path:
        return ROOT / f"eval.2025{self.tag}.scores.csv"

    @property
    def per_response(self) -> Path:
        return ROOT / f"eval.2025{self.tag}"

    @property
    def results_md(self) -> Path:
        return VQA / f"EVALUATION.2025{self.tag}.md"


MODELS = ("gemma4", "qwen3vl", "cosmos")


# ---------------------------------------------------------------------------
# Shell plumbing
# ---------------------------------------------------------------------------

def to_wsl(path: Path) -> str:
    """E:\\TREC_2026\\x -> /mnt/e/TREC_2026/x"""
    resolved = Path(path).resolve()
    drive = resolved.drive.rstrip(":").lower()

    return "/mnt/" + drive + "/" + "/".join(resolved.parts[1:])


@dataclass
class Step:
    """One command, plus what it needs to be checked and logged."""
    argv: list[str]
    host: str = "windows"          # "windows" | "wsl"
    script: Path | None = None     # for the flag-coverage check
    log: str = "step"
    ok_returncodes: tuple[int, ...] = (0,)

    def display(self) -> str:
        if self.host == "wsl":
            return f"wsl[{WSL_DISTRO}] " + " ".join(shlex.quote(a)
                                                    for a in self.argv)

        return subprocess.list2cmdline(self.argv)


def wsl_bash(command: str) -> list[str]:
    return ["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc", command]


def run_step(step: Step, ctx: Ctx) -> int:
    """Execute one step, tee'ing its output to the console and a log file."""
    if step.host == "wsl":
        body = " ".join(shlex.quote(a) for a in step.argv)
        argv = wsl_bash(f"cd {shlex.quote(WSL_ROOT)} && {body}")
    else:
        argv = step.argv

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.log}.log"

    print(f"    $ {step.display()}")

    if ctx.dry_run:
        return 0

    started = time.monotonic()

    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {datetime.now(timezone.utc).isoformat()} "
                  f"{step.display()}\n")
        log.flush()

        process = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            sys.stdout.write("    | " + line)
            log.write(line)

        process.wait()

    elapsed = time.monotonic() - started
    print(f"    ({elapsed / 60:.1f} min, exit {process.returncode})")

    return process.returncode


def quiet(argv: list[str], timeout: float = 120.0) -> tuple[int, str]:
    """Run something whose output we only want to inspect."""
    try:
        done = subprocess.run(argv, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8",
                              errors="replace")
        return done.returncode, (done.stdout or "") + (done.stderr or "")
    except (subprocess.SubprocessError, OSError) as exc:
        return 1, str(exc)


def launch_detached(argv: list[str], timeout: float = 120.0) -> int:
    """Start a long-lived server and genuinely let go of it.

    This must NOT capture stdout/stderr, and the difference is not cosmetic.
    With pipes attached, WSL tears the launched process down as soon as Python
    closes them: the launch returns rc=0, the shell's redirect truncates the
    server log to zero bytes, and nothing is ever listening -- a completely
    silent failure that only shows up as a startup timeout 15 minutes later.

    Verified empirically: the identical command line survives with stdio at
    DEVNULL and dies with capture_output=True. ``setsid`` in the command is
    necessary too, but it is not sufficient on its own.
    """
    try:
        return subprocess.run(argv, stdin=subprocess.DEVNULL,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL,
                              timeout=timeout).returncode
    except (subprocess.SubprocessError, OSError):
        return 1


# ---------------------------------------------------------------------------
# GPU / server arbitration
# ---------------------------------------------------------------------------

def gpu_used_mib() -> int | None:
    code, out = quiet(["nvidia-smi", "--query-gpu=memory.used",
                       "--format=csv,noheader,nounits"], timeout=30)

    if code != 0:
        return None

    for line in out.splitlines():
        if line.strip().isdigit():
            return int(line.strip())

    return None


def http_json(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def http_up(url: str) -> bool:
    try:
        http_json(url, timeout=5.0)
        return True
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


def wait_http(url: str, label: str, timeout: float) -> bool:
    print(f"  waiting for {label} at {url} (up to {timeout / 60:.0f} min)")
    deadline = time.monotonic() + timeout
    last = 0.0

    while time.monotonic() < deadline:
        if http_up(url):
            print(f"  {label} is up")
            return True

        if time.monotonic() - last > 60:
            last = time.monotonic()
            used = gpu_used_mib()
            print(f"    still starting… "
                  f"{int(deadline - time.monotonic())}s left"
                  f"{f', {used} MiB on the GPU' if used else ''}")

        time.sleep(5)

    print(f"  TIMEOUT waiting for {label}", file=sys.stderr)
    return False


def gpu_holders() -> str:
    """Which processes still have memory on the card -- for diagnosing a
    teardown that did not actually free it."""
    code, out = quiet(["nvidia-smi", "--query-compute-apps=pid,process_name,"
                       "used_memory", "--format=csv,noheader"], timeout=30)

    return out.strip() if code == 0 else "(nvidia-smi unavailable)"


def wait_vram(below_mib: int, timeout: float) -> bool:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        used = gpu_used_mib()

        if used is None or used < below_mib:
            return True

        time.sleep(5)

    print(f"  GPU still holds {gpu_used_mib()} MiB after {timeout:.0f}s, "
          f"held by:\n    " + gpu_holders().replace("\n", "\n    "),
          file=sys.stderr)

    return False


# Ollama does not host the model in ollama.exe. It spawns a child --
# lib\ollama\llama-server.exe on current builds, ollama_llama_server.exe on
# older ones -- and that child is what owns the VRAM. Killing only the parent
# closes port 11434 while leaving ~9 GB on the card, which then starves vLLM
# at --gpu-memory-utilization 0.92 and the server never starts. Kill the tree.
OLLAMA_IMAGES = ("ollama app.exe", "ollama.exe", "llama-server.exe",
                 "ollama_llama_server.exe")


def stop_ollama(ctx: Ctx) -> None:
    """Unload the model politely, then make sure nothing survives holding VRAM."""
    if http_up(f"{OLLAMA_URL}/api/tags"):
        # keep_alive 0 tells the server to drop the model now; far cleaner than
        # killing it out from under a loaded 7.4 GB checkpoint.
        payload = json.dumps({"model": GEMMA_MODEL, "messages": [],
                              "keep_alive": 0, "stream": False}).encode()
        request = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat", data=payload,
            headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(request, timeout=120):
                pass
        except (urllib.error.URLError, OSError):
            pass

        if OLLAMA_EXE.exists():
            quiet([str(OLLAMA_EXE), "stop", GEMMA_MODEL], timeout=120)

        time.sleep(3)

    for image in OLLAMA_IMAGES:
        quiet(["taskkill", "/F", "/T", "/IM", image], timeout=60)


def teardown(ctx: Ctx) -> bool:
    """Stop every backend and confirm the card is actually free.

    Order matters: the Cosmos systemd unit has Restart=on-failure, so killing
    its process before stopping the unit just makes systemd bring it back.
    """
    print("  stopping all backends")

    if ctx.dry_run:
        return True

    # '[v]llm serve' rather than 'vllm serve': the pattern is matched against
    # every command line including this bash -lc invocation's own, which
    # contains the literal string -- so a naive pattern makes pkill kill its
    # own shell and abandon the rest of the teardown.
    quiet(wsl_bash(
        "systemctl --user stop cosmos-reason2 2>/dev/null; "
        "systemctl --user stop gemma4-vllm 2>/dev/null; "
        "pkill -f '[v]llm serve' 2>/dev/null; "
        "sleep 2; exit 0"
    ), timeout=180)

    stop_ollama(ctx)

    if wait_vram(2048, 180):
        return True

    # One more pass: something outlived its parent. Retry the tree kill before
    # giving up, since this is exactly the orphan case OLLAMA_IMAGES documents.
    print("  retrying the teardown", file=sys.stderr)

    for image in OLLAMA_IMAGES:
        quiet(["taskkill", "/F", "/T", "/IM", image], timeout=60)

    quiet(wsl_bash("pkill -9 -f '[v]llm serve' 2>/dev/null; exit 0"),
          timeout=120)

    return wait_vram(2048, 120)


def start_ollama(ctx: Ctx) -> bool:
    if http_up(f"{OLLAMA_URL}/api/tags"):
        print("  ollama already up")
    else:
        print(f"  starting {OLLAMA_EXE}")

        if ctx.dry_run:
            return True

        if not OLLAMA_EXE.exists():
            print(f"  no ollama at {OLLAMA_EXE}", file=sys.stderr)
            return False

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        handle = (LOG_DIR / "ollama.server.log").open("a", encoding="utf-8")

        subprocess.Popen(
            [str(OLLAMA_EXE), "serve"], stdout=handle, stderr=handle,
            creationflags=getattr(subprocess, "DETACHED_PROCESS", 0)
            | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
        )

        if not wait_http(f"{OLLAMA_URL}/api/tags", "ollama", 300):
            return False

    if ctx.dry_run:
        return True

    # Warm the model so the first real call is not also a 7.4 GB load.
    print(f"  warming {GEMMA_MODEL}")
    payload = json.dumps({
        "model": GEMMA_MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False, "keep_alive": "30m",
        "options": {"num_predict": 1},
    }).encode()

    request = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat", data=payload,
        headers={"Content-Type": "application/json"})

    try:
        with urllib.request.urlopen(request, timeout=600):
            pass
    except (urllib.error.URLError, OSError) as exc:
        print(f"  could not warm {GEMMA_MODEL}: {exc}", file=sys.stderr)
        return False

    return True


def start_gemma(ctx: Ctx) -> bool:
    """Bring up the Gemma 4 vLLM server, systemd first as with Cosmos."""
    if http_up(f"{GEMMA_URL}/models"):
        print("  gemma4 server already up")
        return True

    print("  starting the Gemma 4 vLLM server in WSL2")

    if ctx.dry_run:
        return True

    quiet(wsl_bash("systemctl --user start gemma4-vllm"), timeout=120)

    if wait_http(f"{GEMMA_URL}/models", "gemma4 (systemd)", 420):
        return True

    print("  systemd unit did not come up -- falling back to a direct launch")
    launch_detached(wsl_bash(GEMMA_SERVE), timeout=120)

    return wait_http(f"{GEMMA_URL}/models", "gemma4", 600)


def start_qwen(ctx: Ctx) -> bool:
    if http_up(f"{QWEN_URL}/models"):
        print("  qwen3vl server already up")
        return True

    print("  starting the Qwen3-VL vLLM server in WSL2")

    if ctx.dry_run:
        return True

    launch_detached(wsl_bash(QWEN_SERVE), timeout=120)

    return wait_http(f"{QWEN_URL}/models", "qwen3vl", 900)


def start_cosmos(ctx: Ctx) -> bool:
    if http_up(f"{COSMOS_URL}/models"):
        print("  cosmos server already up")
        return True

    print("  starting the Cosmos-Reason2 vLLM server in WSL2")

    if ctx.dry_run:
        return True

    quiet(wsl_bash("systemctl --user start cosmos-reason2"), timeout=120)

    if wait_http(f"{COSMOS_URL}/models", "cosmos (systemd)", 420):
        return True

    print("  systemd unit did not come up -- falling back to a direct launch")
    launch_detached(wsl_bash(COSMOS_SERVE), timeout=120)

    return wait_http(f"{COSMOS_URL}/models", "cosmos", 600)


_backend = {"current": None}


def ensure_backend(kind: str, ctx: Ctx) -> bool:
    """The only function allowed to start or stop a model server.

    ``BACKEND_ANY`` is a deliberate no-op: a CPU-only stage must leave whatever
    is on the card alone, because the 2026 run's server may still be serving it.
    """
    if kind == BACKEND_ANY or _backend["current"] == kind:
        return True

    # A healthy server for the backend we want is worth adopting rather than
    # restarting: on a resume ``current`` is None, and tearing down a server
    # that is already serving costs several minutes of reload for nothing.
    health = {BACKEND_QWEN: f"{QWEN_URL}/models",
              BACKEND_COSMOS: f"{COSMOS_URL}/models",
              BACKEND_GEMMA: f"{GEMMA_URL}/models",
              BACKEND_OLLAMA: f"{OLLAMA_URL}/api/tags"}.get(kind)

    if health and _backend["current"] is None and http_up(health):
        print(f"\n  [backend] adopting the running {kind} server")
        _backend["current"] = kind
        return True

    print(f"\n  [backend] {_backend['current'] or 'unknown'} -> {kind}")

    if not teardown(ctx):
        # Starting a server on a card that is still occupied just burns the
        # startup timeout and fails opaquely 15 minutes later.
        print("  the GPU was not freed -- refusing to start a server that "
              "cannot allocate", file=sys.stderr)
        return False

    ok = True

    if kind == BACKEND_OLLAMA:
        ok = start_ollama(ctx)
    elif kind == BACKEND_QWEN:
        ok = start_qwen(ctx)
    elif kind == BACKEND_COSMOS:
        ok = start_cosmos(ctx)
    elif kind == BACKEND_GEMMA:
        ok = start_gemma(ctx)

    if ok:
        _backend["current"] = kind

    return ok


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------

def fingerprint() -> dict:
    """Size + sha256 of every guarded 2026 artifact, plus the media inventory."""
    snapshot: dict[str, dict] = {}

    for name in GUARDED:
        path = ROOT / name

        if not path.exists():
            snapshot[name] = {"present": False}
            continue

        digest = hashlib.sha256()

        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)

        snapshot[name] = {"present": True, "size": path.stat().st_size,
                          "sha256": digest.hexdigest()}

    media = ROOT / "videos_TEST_VQA" / "videos"

    if media.is_dir():
        files = [p for p in media.iterdir() if p.is_file()]
        snapshot["videos_TEST_VQA/videos"] = {
            "present": True, "n_files": len(files),
            "total_bytes": sum(p.stat().st_size for p in files),
        }

    return snapshot


def verify(baseline: dict, stage: str) -> bool:
    current = fingerprint()
    changed = [name for name in baseline if baseline[name] != current.get(name)]

    if changed:
        print(f"\nABORT: stage '{stage}' modified 2026 artifacts: {changed}",
              file=sys.stderr)
        print("  A 2025 stage was pointed at a 2026 default. Restore from "
              "backup before continuing.", file=sys.stderr)
        return False

    return True


def add_argument_calls(source: str) -> list[tuple[str, str]]:
    """Every ``<receiver>.add_argument(...)`` call, paren-balanced.

    The receiver is kept because a script with subcommands
    (``sample.2025.eval.py``) declares one group of flags per subparser, and
    only the group a given step actually invokes is relevant to it.
    """
    calls: list[tuple[str, str]] = []

    for match in re.finditer(r"(\w+)\.add_argument\(", source):
        depth = 0
        start = match.end() - 1

        for index in range(start, len(source)):
            if source[index] == "(":
                depth += 1
            elif source[index] == ")":
                depth -= 1

                if depth == 0:
                    calls.append((match.group(1), source[start:index + 1]))
                    break

    return calls


def backend_path_flags(source: str) -> set[str]:
    """Flags a BACKENDS table defaults to a ``ROOT /``-relative path.

    ``--backend vllm`` and ``--backend ollama`` write to different files, so
    the Gemma 4 scripts resolve --out and --cache from the chosen backend
    rather than from a ``DEFAULT_* = ROOT / ...`` constant. That puts them out
    of reach of the rule in path_flags() while leaving them just as able to
    point a 2025 stage at a 2026 artifact.

    A script carrying a BACKENDS table whose resolution block this cannot read
    raises rather than returning nothing: failing open here is precisely how a
    2025 run ends up overwriting `gemma4.run.task1.json`.
    """
    if not re.search(r"^BACKENDS\s*=\s*\{", source, re.MULTILINE):
        return set()

    # A BACKENDS table that defaults no paths at all is fine and common --
    # expand.answers.py has one, but its --out and --cache are required=True,
    # so there is nothing here to leave at a default.
    if "ROOT / defaults[" not in source:
        return set()

    match = re.search(
        r"for name in \(([^)]+)\):\s*\n"
        r"\s*if getattr\(args, name\) is None:\s*\n"
        r"\s*setattr\(args, name, ROOT / defaults\[name\]\)",
        source)

    if not match:
        raise ValueError(
            "carries a BACKENDS table but no readable "
            "'ROOT / defaults[name]' resolution block -- teach "
            "backend_path_flags() its shape rather than skipping it")

    return {"--" + name.strip().strip("\"'").replace("_", "-")
            for name in match.group(1).split(",") if name.strip()}


def path_flags(script: Path) -> dict[str, set[str]]:
    """Receiver -> flags whose default is a ``DEFAULT_* = ROOT / ...`` constant.

    These are exactly the arguments that would silently point a 2025 stage at a
    2026 artifact if left off the command line.
    """
    try:
        source = script.read_text(encoding="utf-8")
    except OSError:
        return {}

    path_consts = set(re.findall(r"^(DEFAULT_[A-Z0-9_]+)\s*=\s*ROOT\s*/",
                                 source, re.MULTILINE))

    if not path_consts:
        return {}

    groups: dict[str, set[str]] = {}

    for receiver, call in add_argument_calls(source):
        default = re.search(r"default\s*=\s*(DEFAULT_[A-Z0-9_]+)", call)

        if not default or default.group(1) not in path_consts:
            continue

        names = re.findall(r'"(--[a-zA-Z0-9][a-zA-Z0-9-]*)"', call)

        if names:
            groups.setdefault(receiver, set()).add(names[0])

    # The Gemma 4 scripts resolve --out and --cache from whichever --backend
    # was chosen instead of from a DEFAULT_* constant, so the loop above cannot
    # see them. They are exactly as dangerous to leave off.
    from_backends = backend_path_flags(source)

    if from_backends:
        for receiver in groups or {"ap"}:
            groups.setdefault(receiver, set()).update(from_backends)

    return groups


def check_flags(steps: list[Step]) -> list[str]:
    """Every path-valued flag of every driven script must be given explicitly."""
    problems: list[str] = []

    for step in steps:
        if not step.script or not step.script.exists():
            continue

        groups = path_flags(step.script)
        given = {a for a in step.argv if a.startswith("--")}

        for receiver, required in groups.items():
            # With subcommands, skip the groups belonging to the subparsers
            # this step does not invoke.
            if len(groups) > 1 and not (required & given):
                continue

            missing = sorted(required - given)

            if missing:
                problems.append(f"{step.script.name} [{receiver}]: "
                                f"{', '.join(missing)} left at its default")

    return problems


# ---------------------------------------------------------------------------
# Stage bodies
# ---------------------------------------------------------------------------

def py(script: Path, *args: str) -> list[str]:
    exe = str(WINDOWS_PYTHON) if WINDOWS_PYTHON.exists() else sys.executable
    return [exe, str(script), *args]


def wsl_py(script: Path, *args: str) -> list[str]:
    return [WSL_PYTHON, f"VQA/{script.name}", *args]


def steps_queries(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "sample.2025.eval.py", "convert",
           "--gt", str(ctx.gt_full),
           "--videos", str(VIDEOS),
           "--queries-out", str(ctx.queries_all),
           "--pool-out", str(ctx.pool),
           "--test-2026", str(QUERIES_2026)),
        script=VQA / "sample.2025.eval.py", log="queries",
    )]


def steps_durations(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "duration.report.py",
           "--queries", str(ctx.queries_all),
           "--videos", str(VIDEOS),
           "--report", str(ctx.durations),
           "--html", str(ctx.durations_html)),
        script=VQA / "duration.report.py", log="durations",
    )]


def steps_sample(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "sample.2025.eval.py", "select",
           "--pool", str(ctx.pool),
           "--durations", str(ctx.durations),
           "--metadata", str(METADATA),
           "-n", str(ctx.n),
           "--seed", str(ctx.seed),
           "--out", str(ctx.queries),
           "--gt-out", str(ctx.gt),
           "--reserves-out", str(ctx.reserves),
           "--report-json", str(ctx.sample_report),
           "--report-md", str(ctx.sample_report_md)),
        script=VQA / "sample.2025.eval.py", log="sample",
    )]


def steps_frames(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "extract.frames.py",
           "--queries", str(ctx.queries),
           "--videos", str(VIDEOS),
           "--durations", str(ctx.durations),
           "--out", str(FRAMES)),
        script=VQA / "extract.frames.py", log="frames",
    )]


def sample_video_ids(ctx: Ctx) -> list[str]:
    if not ctx.queries.exists():
        return []

    with ctx.queries.open("r", encoding="utf-8-sig") as handle:
        return sorted({Path(q["Video_ID"]).stem for q in json.load(handle)})


def steps_whisper(ctx: Ctx) -> list[Step]:
    ids = sample_video_ids(ctx)

    return [Step(
        py(VQA / "transcribe.whisper.py",
           "--videos-dir", str(VIDEOS),
           "--out-dir", str(TRANSCRIPTS),
           "--model-dir", str(ROOT / "models" / "faster-whisper-large-v3"),
           *(["--only", *ids] if ids else [])),
        script=VQA / "transcribe.whisper.py", log="whisper",
    )]


# The three Gemma 4 stages run from Windows against the vLLM server in WSL2.
# That crossing is fine here and only here: these clients send their frames as
# base64 in the request body, so nothing depends on the server being able to
# open a path. The Qwen3-VL and Cosmos clients hand over a file:// URI, which
# is why they must run inside WSL2. Confirmed reachable: Windows localhost:8010
# forwards into the distro.
def steps_route(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "route.query.inputs.py",
           "--queries", str(ctx.queries),
           "--out", str(ctx.routing),
           "--cache", str(ctx.routing_cache),
           *ctx.gemma_flags()),
        script=VQA / "route.query.inputs.py", log="route",
    )]


def steps_variations(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "generate.query.variations.py",
           "--queries", str(ctx.queries),
           "--out", str(ctx.variations),
           "--cache", str(ctx.variations_cache),
           *ctx.gemma_flags()),
        script=VQA / "generate.query.variations.py", log="variations",
    )]


def steps_gemma4(ctx: Ctx) -> list[Step]:
    return [Step(
        py(VQA / "generate.answers.py",
           "--queries", str(ctx.queries),
           "--variations", str(ctx.variations),
           "--frames", str(FRAMES),
           "--transcripts", str(TRANSCRIPTS),
           "--out", str(ctx.run_file("gemma4")),
           "--cache", str(ctx.run_cache("gemma4")),
           "--workers", str(ctx.workers_gemma),
           *ctx.gemma_flags()),
        script=VQA / "generate.answers.py", log="answers.gemma4",
    )]


def steps_qwen3vl(ctx: Ctx) -> list[Step]:
    return [Step(
        wsl_py(VQA / "generate.answers.qwen3vl.py",
               "--queries", to_wsl(ctx.queries),
               "--routing", to_wsl(ctx.routing),
               "--videos", to_wsl(VIDEOS),
               "--frame-index", to_wsl(FRAMES / "index.json"),
               "--transcripts", to_wsl(TRANSCRIPTS),
               "--out", to_wsl(ctx.run_file("qwen3vl")),
               "--cache", to_wsl(ctx.run_cache("qwen3vl")),
               "--workers", str(ctx.workers_qwen)),
        host="wsl", script=VQA / "generate.answers.qwen3vl.py",
        log="answers.qwen3vl",
    )]


def steps_cosmos(ctx: Ctx) -> list[Step]:
    return [Step(
        wsl_py(VQA / "generate.answers.cosmos.py",
               "--queries", to_wsl(ctx.queries),
               "--videos", to_wsl(VIDEOS),
               "--frame-index", to_wsl(FRAMES / "index.json"),
               "--transcripts", to_wsl(TRANSCRIPTS),
               "--out", to_wsl(ctx.run_file("cosmos")),
               "--cache", to_wsl(ctx.run_cache("cosmos")),
               "--workers", str(ctx.workers_cosmos)),
        host="wsl", script=VQA / "generate.answers.cosmos.py",
        log="answers.cosmos",
    )]


def steps_validate(ctx: Ctx) -> list[Step]:
    return [
        Step(py(VQA / "validate.ag.run.py",
                str(ctx.queries), str(ctx.run_file(model)),
                "--report-json", str(ctx.validation(model))),
             script=VQA / "validate.ag.run.py", log=f"validate.{model}",
             # A run is still worth scoring when the validator only warns.
             ok_returncodes=(0, 1))
        for model in MODELS if ctx.dry_run or ctx.run_file(model).exists()
    ]


def steps_score(ctx: Ctx) -> list[Step]:
    runs: list[str] = []

    for model in MODELS:
        if ctx.dry_run or ctx.run_file(model).exists():
            runs += ["--run", f"{model}={ctx.run_file(model)}"]

    return [Step(
        py(VQA / "score.answers.py",
           "--gt", str(ctx.gt),
           *runs,
           "--metrics", ctx.metrics,
           "--out", str(ctx.scores),
           "--csv", str(ctx.scores_csv),
           "--per-response-dir", str(ctx.per_response),
           "--oracle"),
        script=VQA / "score.answers.py", log="score",
    )]


# -- stages that are python, not a subprocess --------------------------------

def qwen3vl_2026_state(total: int) -> dict:
    """Is the 2026 Qwen3-VL run *finished*, in every sense that matters?

    Three independent conditions, because any one alone can lie:

    - the cache holds every Q_ID (it is appended as work completes);
    - ``qwen3vl.run.task1.json`` exists and carries every Q_ID -- the script
      writes it only after the inference loop returns normally, so a Ctrl-C'd
      run (which exits 130 before ``assemble()``) leaves the cache full-looking
      but no run file;
    - no ``generate.answers.qwen3vl.py`` process is alive in WSL, so the last
      writes have actually landed and nothing will append behind our back.
    """
    cached = count_cached(QWEN3VL_2026_CACHE)
    run_path = ROOT / "qwen3vl.run.task1.json"
    run_entries = 0

    if run_path.exists():
        try:
            run_entries = len(json.loads(run_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError, ValueError):
            run_entries = 0

    code, out = quiet(wsl_bash(
        "pgrep -fa 'generate\\.answers\\.qwen3vl\\.py' || true"), timeout=120)
    alive = bool(out.strip()) and code == 0

    return {
        "cached": cached,
        "run_entries": run_entries,
        "run_file": run_path.name,
        "process_alive": alive,
        "done": cached >= total and run_entries >= total and not alive,
    }


def stage_gate(ctx: Ctx, args: argparse.Namespace) -> bool:
    """Do not touch the GPU until the 2026 Qwen3-VL run has finished."""
    if args.no_gate:
        print("  --no-gate: skipping the 2026-run check")
        return True

    total = len(load_queries(QUERIES_2026))
    deadline = time.monotonic() + args.wait_for_2026
    last_report = 0.0
    previous = None

    while True:
        state = qwen3vl_2026_state(total)

        if state["done"]:
            print(f"  2026 qwen3vl run is complete: {state['cached']}/{total} "
                  f"cached, {state['run_entries']}/{total} in "
                  f"{state['run_file']}, process exited")
            return True

        # Report on every change, and at least every 15 minutes so a long wait
        # still shows a heartbeat in the log.
        key = (state["cached"], state["run_entries"], state["process_alive"])

        if key != previous or time.monotonic() - last_report > 900:
            previous = key
            last_report = time.monotonic()

            blockers = []

            if state["cached"] < total:
                blockers.append(f"cache {state['cached']}/{total}")
            if state["run_entries"] < total:
                blockers.append(
                    f"{state['run_file']} "
                    f"{state['run_entries'] or 'not written'}/{total}")
            if state["process_alive"]:
                blockers.append("inference still running")

            print(f"  [{datetime.now().strftime('%H:%M:%S')}] waiting on the "
                  f"2026 run: {'; '.join(blockers)}")

        if ctx.dry_run or time.monotonic() >= deadline:
            break

        time.sleep(300)

    if ctx.dry_run:
        return True

    state = qwen3vl_2026_state(total)
    print(f"\n  the 2026 run has not finished after "
          f"{args.wait_for_2026 / 3600:.1f} h "
          f"(cache {state['cached']}/{total}, run file "
          f"{state['run_entries']}/{total}, "
          f"{'still running' if state['process_alive'] else 'not running'}).\n"
          f"  If it died, finish it first; --no-gate proceeds anyway and will "
          f"stop its server.", file=sys.stderr)

    return False


def load_queries(path: Path) -> list:
    if not path.exists():
        return []

    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def count_cached(path: Path) -> int:
    if not path.exists():
        return 0

    seen: set[int] = set()

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if not line:
                continue

            try:
                seen.add(int(json.loads(line)["Q_ID"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue

    return len(seen)


def stage_deps(ctx: Ctx, args: argparse.Namespace) -> bool:
    ok = True

    print("  binaries")

    for name, argv in (("ffmpeg", ["ffmpeg", "-version"]),
                       ("ffprobe", ["ffprobe", "-version"]),
                       ("wsl.exe", ["wsl.exe", "-l", "-q"]),
                       ("nvidia-smi", ["nvidia-smi", "-L"])):
        code, _ = quiet(argv, timeout=60)
        print(f"    {name:<12} {'ok' if code == 0 else 'MISSING'}")
        ok &= code == 0

    print(f"    {'ollama':<12} "
          f"{'ok' if OLLAMA_EXE.exists() else 'MISSING'}")
    ok &= OLLAMA_EXE.exists()

    print("  wsl python + openai")
    code, out = quiet(wsl_bash(
        f"{WSL_PYTHON} -c 'import openai, sys; print(openai.__version__)'"
    ), timeout=180)
    print(f"    {'openai':<12} {'ok ' + out.strip() if code == 0 else 'MISSING'}")
    ok &= code == 0

    print("  windows scorer packages")
    exe = str(WINDOWS_PYTHON) if WINDOWS_PYTHON.exists() else sys.executable
    missing: list[str] = []

    for module, package in SCORER_PACKAGES:
        code, _ = quiet([exe, "-c", f"import {module}"], timeout=180)
        print(f"    {module:<12} {'ok' if code == 0 else 'MISSING'}")

        if code != 0:
            missing.append(package)

    if missing:
        if args.install_deps and not ctx.dry_run:
            print(f"  installing {' '.join(missing)}")
            code, out = quiet([exe, "-m", "pip", "install", *missing],
                              timeout=1800)
            print(out[-2000:])
            ok &= code == 0
        else:
            print(f"\n  install them with:\n    {exe} -m pip install "
                  f"{' '.join(missing)}\n  or re-run with --install-deps",
                  file=sys.stderr)
            ok = False

    print("  path-flag coverage")
    problems = check_flags(all_steps(ctx))

    if problems:
        for problem in problems:
            print(f"    FAIL {problem}", file=sys.stderr)
        ok = False
    else:
        print("    every path flag is passed explicitly")

    if not VIDEOS.is_dir():
        print(f"    FAIL no videos at {VIDEOS}", file=sys.stderr)
        ok = False

    return ok


def stage_smoke_sample(ctx: Ctx, args: argparse.Namespace) -> bool:
    """Carve the smoke set out of the real sample, deterministically."""
    full = Ctx(n=args.n, smoke=False, seed=ctx.seed, dry_run=ctx.dry_run,
               force=ctx.force, workers_qwen=ctx.workers_qwen,
               workers_cosmos=ctx.workers_cosmos,
               workers_gemma=ctx.workers_gemma,
               gemma_backend=ctx.gemma_backend, metrics=ctx.metrics)

    if not full.queries.exists():
        print(f"  no {full.queries.name} -- run the sample stage first",
              file=sys.stderr)
        return False

    queries = load_queries(full.queries)[:ctx.n]
    wanted = {q["Q_ID"] for q in queries}

    if ctx.dry_run:
        print(f"  would carve {len(queries)} queries into {ctx.queries.name} "
              f"and {ctx.gt.name}")
        return True

    with ctx.queries.open("w", encoding="utf-8") as handle:
        json.dump(queries, handle, indent=4, ensure_ascii=False)
        handle.write("\n")

    rows = [line for line in full.gt.read_text(encoding="utf-8").splitlines()
            if line.strip() and int(line.split(",", 1)[0]) in wanted]
    ctx.gt.write_text("\n".join(rows) + "\n", encoding="utf-8")

    print(f"  smoke set: {len(queries)} queries -> {ctx.queries.name}, "
          f"{len(rows)} GT rows -> {ctx.gt.name}")

    return True


def stage_report(ctx: Ctx, args: argparse.Namespace) -> bool:
    """Write the results document."""
    if ctx.dry_run:
        print(f"  would write {ctx.results_md}")
        return True

    if not ctx.scores.exists():
        print(f"  no {ctx.scores.name} -- nothing to report", file=sys.stderr)
        return False

    scores = json.loads(ctx.scores.read_text(encoding="utf-8"))
    sample = json.loads(ctx.sample_report.read_text(encoding="utf-8")) \
        if ctx.sample_report.exists() else {}

    columns = sorted({key for run in scores["runs"].values()
                      for key in run
                      if key in ("METEOR", "BERTScore", "STS_local",
                                 "NDCG_METEOR", "NDCG_BERTScore")})

    lines = [
        f"# 2025 ground-truth evaluation ({ctx.label})",
        "",
        f"Generated {scores['generated']}. Ground truth `{ctx.gt.name}`, "
        f"{scores['gt_rows']} queries.",
        "",
        "## Results",
        "",
        "| Run | Queries | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 2) + "|",
    ]

    for name, run in scores["runs"].items():
        if name == "__oracle__":
            continue

        lines.append(f"| {name} | {run['n_queries']} | " +
                     " | ".join(f"{run.get(c, 0.0):.3f}" for c in columns) +
                     " |")

    if "__oracle__" in scores["runs"]:
        oracle = scores["runs"]["__oracle__"]
        lines += ["",
                  f"Oracle (ground truth scored against itself): " +
                  ", ".join(f"{c} {oracle.get(c, 0.0):.3f}" for c in columns) +
                  ". Values near 1.0 confirm normalisation and alignment."]

    if scores.get("comparisons"):
        lines += ["", "## Paired comparisons", "",
                  "| Metric | Baseline | Candidate | Mean Δ | Better | p |",
                  "|---|---|---|---|---|---|"]

        for row in scores["comparisons"]:
            lines.append(
                f"| {row['metric']} | {row['baseline']} | {row['candidate']} | "
                f"{row['mean_delta']:+.4f} | "
                f"{row['candidate_better']}/{row['n']} | "
                f"{row['wilcoxon_p'] if row['wilcoxon_p'] is not None else 'n/a'} |"
            )

    if sample:
        lines += ["", "## Sample", "",
                  f"{sample['drawn']} of {sample['requested']} requested, "
                  f"seed `{sample['seed']}`, drawn from a pool of "
                  f"{sample['pool_size']}.", "",
                  "| Bin | Pool | Drawn | 2026 target |", "|---|---|---|---|"]

        for block in sample["bins"]:
            if not block["pool"] and not block["target_2026_count"]:
                continue

            lines.append(f"| {block['bin']} | {block['pool']} | "
                         f"{block['achieved']} | {block['target_2026_count']} |")

    lines += [
        "",
        "## What these numbers are, and are not",
        "",
        "- **Quote them as \"on a 150-query 2025 sample\", never as \"our "
        "METEOR\".** Per-query spread is far wider than the differences "
        "between runs; the absolute figure carries roughly ±0.03.",
        "- **Short video only.** Every 2025 ground-truth clip is a YouTube "
        "Short (5–30 s, median 15 s). On this corpus "
        "`adaptive_fps = min(4.0, max(0.2, 64 / duration))` clamps to 2.1–4.0 "
        "fps for every clip, and `LONG_VIDEO_SEC = 60` puts all of them on "
        "Gemma 4's 16-frame `CAP_SHORT` path. The fps floor, the frame-token "
        "degrade ladder, the 32-frame long path and the 4 GiB decode-limit "
        "transcode are never exercised. The 2026 test set runs to 389 s with "
        "6.3% over 150 s, and nothing here measures that.",
        "- **The pool is 1,327 of 2,000.** The GT download was interrupted at "
        "Q_ID 1471, so the sample is drawn from a positional slice of the "
        "ground truth rather than the whole of it.",
        "- **Mild style leakage.** `MAX_ANSWER_WORDS = 40` and the shared "
        "\"most good answers are under ten words\" prompt rule were tuned "
        "against this very ground truth's length distribution "
        "(`generate.answers.qwen3vl.py:269-273`). That flatters the "
        "length-sensitive part of METEOR. It cannot be removed without "
        "changing the pipeline under test.",
    ]

    if scores.get("sts_note"):
        lines += ["- **STS_local is not the official STS.** "
                  + scores["sts_note"]]

    lines += [
        "",
        "## Configuration under test",
        "",
        "The corpus-derived tuning constants were left at their 2026 values "
        "deliberately — retuning them for short clips would mean the number "
        "produced is not the 2026 pipeline's number:",
        "",
        "| Constant | Value | Where |",
        "|---|---|---|",
        "| `TARGET_FRAMES` | 64 | qwen3vl, cosmos |",
        "| `DENSE_TARGET_FRAMES` | 96 | qwen3vl |",
        "| `MAX_FPS` / `MIN_FPS` | 4.0 / 0.2 | qwen3vl, cosmos |",
        "| `MIN`/`MAX_FRAME_TOKENS` | 128 / 768 | qwen3vl |",
        "| `TEXT_TOKEN_RESERVE` | 2048 | qwen3vl |",
        "| `VISUAL_BUDGET_SAFETY` | 0.8 | qwen3vl |",
        "| `MIN_FRAMES`/`CAP_SHORT`/`CAP_LONG` | 8 / 16 / 32 | extract.frames |",
        "| `LONG_VIDEO_SEC` | 60.0 | extract.frames |",
        "",
    ]

    # This file accumulates hand-written analysis that no generator can
    # reproduce (the appendix documenting defects, guards and commands). The
    # stage marker normally skips regeneration, but --force would overwrite it,
    # so keep a copy rather than destroying the only record.
    if ctx.results_md.exists():
        backup = ctx.results_md.with_suffix(".md.bak")
        backup.write_text(ctx.results_md.read_text(encoding="utf-8"),
                          encoding="utf-8")
        print(f"  existing report preserved as {backup.name}")

    ctx.results_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {ctx.results_md}")

    return True


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    backend: str
    build: object = None            # (ctx) -> list[Step]
    body: object = None             # (ctx, args) -> bool
    marker: object = None           # (ctx) -> Path | None
    smoke_only: bool = False
    real_only: bool = False
    description: str = ""


STAGES: list[Stage] = [
    Stage("gate", BACKEND_ANY, body=stage_gate,
          description="wait for the 2026 qwen3vl run to finish"),
    Stage("deps", BACKEND_ANY, body=stage_deps,
          description="binaries, packages, path-flag coverage"),
    Stage("queries", BACKEND_ANY, build=steps_queries,
          marker=lambda c: c.pool, description="2025 CSV -> query JSON + pool"),
    Stage("durations", BACKEND_ANY, build=steps_durations,
          marker=lambda c: c.durations, description="ffprobe every GT video"),
    Stage("sample", BACKEND_ANY, build=steps_sample,
          marker=lambda c: ROOT / f"vqa.2025.sample{c.n}.json"
          if not c.smoke else None,
          real_only=True, description="stratified draw + GT CSV"),
    Stage("smoke-sample", BACKEND_ANY, body=stage_smoke_sample,
          marker=lambda c: c.queries, smoke_only=True,
          description="carve the rehearsal set out of the sample"),
    Stage("frames", BACKEND_ANY, build=steps_frames,
          marker=lambda c: FRAMES / "index.json",
          description="frames for Gemma 4"),
    Stage("whisper", BACKEND_FREE, build=steps_whisper,
          marker=lambda c: TRANSCRIPTS / "transcription_index.json",
          description="ASR over the sampled clips"),
    Stage("route", BACKEND_GEMMA, build=steps_route,
          marker=lambda c: c.routing, description="routed inputs per query"),
    Stage("variations", BACKEND_GEMMA, build=steps_variations,
          marker=lambda c: c.variations,
          description="paraphrases for Gemma 4's ensemble"),
    Stage("answers:gemma4", BACKEND_GEMMA, build=steps_gemma4,
          marker=lambda c: c.run_file("gemma4"), description="Gemma 4 run"),
    Stage("answers:qwen3vl", BACKEND_QWEN, build=steps_qwen3vl,
          marker=lambda c: c.run_file("qwen3vl"), description="Qwen3-VL run"),
    Stage("answers:cosmos", BACKEND_COSMOS, build=steps_cosmos,
          marker=lambda c: c.run_file("cosmos"), description="Cosmos run"),
    Stage("validate", BACKEND_ANY, build=steps_validate,
          marker=lambda c: None, description="run-format validation"),
    Stage("score", BACKEND_FREE, build=steps_score,
          marker=lambda c: c.scores, description="METEOR / BERTScore / NDCG"),
    Stage("report", BACKEND_ANY, body=stage_report,
          marker=lambda c: c.results_md, description="write EVALUATION.2025.md"),
]


def applicable(ctx: Ctx) -> list[Stage]:
    return [s for s in STAGES
            if not (s.smoke_only and not ctx.smoke)
            and not (s.real_only and ctx.smoke)]


def all_steps(ctx: Ctx) -> list[Step]:
    steps: list[Step] = []

    for stage in applicable(ctx):
        if stage.build:
            try:
                steps.extend(stage.build(ctx))
            except Exception:  # noqa: BLE001 - the check must not crash deps
                continue

    return steps


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {"stages": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=4) + "\n", encoding="utf-8")


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run the 2025 ground-truth evaluation end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + ", ".join(s.name for s in STAGES),
    )

    ap.add_argument("--smoke", type=int, nargs="?", const=5, metavar="N",
                    help="rehearse the whole chain on N queries (default 5) "
                         "with .smoke artifacts")
    ap.add_argument("-n", "--n", type=int, default=150,
                    help="sample size (default: %(default)s)")
    ap.add_argument("--seed", type=int, default=20260814)

    ap.add_argument("--stages", help="comma-separated subset to run")
    ap.add_argument("--from", dest="from_stage", help="start at this stage")
    ap.add_argument("--to", dest="to_stage", help="stop after this stage")

    ap.add_argument("--dry-run", action="store_true",
                    help="print every command, execute nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages whose output already exists")
    ap.add_argument("--no-gate", action="store_true",
                    help="do not wait for the 2026 qwen3vl run (it will be "
                         "stopped when the GPU is needed)")
    ap.add_argument("--wait-for-2026", type=float, default=12 * 3600,
                    help="seconds to wait on the gate (default: 12 h)")
    ap.add_argument("--install-deps", action="store_true",
                    help="pip install the missing scorer packages")
    ap.add_argument("--no-guard", action="store_true",
                    help="skip the 2026-artifact integrity check (unwise)")
    ap.add_argument("--keep-servers", action="store_true",
                    help="leave the last backend running at the end")
    ap.add_argument("--restore-2026", action="store_true",
                    help="relaunch the Qwen3-VL server when everything is done")

    ap.add_argument("--workers-qwen", type=int, default=4)
    ap.add_argument("--workers-cosmos", type=int, default=2)
    # 8 is the measured knee for Gemma 4 on vLLM (VQA/GEMMA4_VLLM_SETUP.md §6);
    # it drops to 1 automatically on the Ollama backend, which has one slot.
    ap.add_argument("--workers-gemma", type=int,
                    help="default: 8 on the vllm backend, 1 on ollama")
    ap.add_argument("--gemma4-backend", choices=("vllm", "ollama"),
                    default="vllm",
                    help="where the three Gemma 4 stages run. vllm is the "
                         "server in WSL2 on 8010; ollama is the old Windows "
                         "path (default: %(default)s)")
    ap.add_argument("--metrics", default="meteor,bert",
                    help="passed to score.answers.py (default: %(default)s)")

    return ap.parse_args(argv)


def select_stages(args: argparse.Namespace, ctx: Ctx) -> list[Stage]:
    stages = applicable(ctx)
    names = [s.name for s in stages]

    if args.stages:
        wanted = [n.strip() for n in args.stages.split(",") if n.strip()]
        unknown = [n for n in wanted if n not in names]

        if unknown:
            raise SystemExit(f"unknown stage(s): {unknown}\nknown: {names}")

        return [s for s in stages if s.name in wanted]

    start = 0
    end = len(stages)

    if args.from_stage:
        if args.from_stage not in names:
            raise SystemExit(f"unknown stage {args.from_stage}\nknown: {names}")
        start = names.index(args.from_stage)

    if args.to_stage:
        if args.to_stage not in names:
            raise SystemExit(f"unknown stage {args.to_stage}\nknown: {names}")
        end = names.index(args.to_stage) + 1

    return stages[start:end]


def main(argv=None) -> int:
    args = parse_args(argv)

    if os.name != "nt":
        print("This orchestrator must run on Windows: it drives the Windows "
              "Ollama install directly and dispatches the vLLM stages through "
              "wsl.exe. See the module docstring.", file=sys.stderr)
        return 1

    ctx = Ctx(
        n=args.smoke if args.smoke else args.n,
        smoke=bool(args.smoke),
        seed=args.seed,
        dry_run=args.dry_run,
        force=args.force,
        workers_qwen=args.workers_qwen,
        workers_cosmos=args.workers_cosmos,
        workers_gemma=args.workers_gemma if args.workers_gemma is not None
        else (1 if args.gemma4_backend == "ollama" else 8),
        gemma_backend=args.gemma4_backend,
        metrics=args.metrics,
    )

    stages = select_stages(args, ctx)

    print(f"TREC 2025 evaluation — {'SMOKE ' if ctx.smoke else ''}"
          f"{ctx.n} queries, seed {ctx.seed}")
    print(f"artifacts: {ctx.queries.name}, {ctx.gt.name}, "
          f"*.run.2025{ctx.tag}.json")
    print(f"stages:    {' -> '.join(s.name for s in stages)}\n")

    state = load_state()

    # The baseline is armed *after* the gate, not before it. The gate exists to
    # wait while the 2026 run finishes, and that run legitimately appends to
    # answers.task1.qwen3vl.cache.jsonl and finally writes qwen3vl.run.task1.json
    # -- both guarded files. Snapshotting first turns the 2026 run's own
    # progress into a tampering report and aborts a five-hour wait at the
    # moment it succeeds.
    baseline = None
    armed = args.no_guard

    started_all = time.monotonic()

    for stage in stages:
        if not armed and stage.name != "gate":
            baseline = fingerprint()
            armed = True
            present = sum(1 for v in baseline.values() if v.get("present"))
            print(f"guarding {present} 2026 artifacts against modification "
                  f"from '{stage.name}' onward\n")

        marker = stage.marker(ctx) if stage.marker else None

        if marker and marker.exists() and not ctx.force:
            print(f"[{stage.name}] already done ({marker.name}) — skipping")
            continue

        print(f"[{stage.name}] {stage.description}")
        started = time.monotonic()

        backend = ctx.backend_for(stage.backend)

        if not ensure_backend(backend, ctx):
            print(f"  could not bring up backend '{backend}'",
                  file=sys.stderr)
            return 1

        ok = True

        if stage.body:
            ok = stage.body(ctx, args)

        if ok and stage.build:
            steps = stage.build(ctx)

            if not steps:
                print("  nothing to do")

            for step in steps:
                code = run_step(step, ctx)

                if code not in step.ok_returncodes:
                    print(f"  FAILED (exit {code})", file=sys.stderr)
                    ok = False
                    break

        elapsed = time.monotonic() - started
        state["stages"][stage.name] = {
            "ok": ok,
            "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "minutes": round(elapsed / 60, 2),
            "smoke": ctx.smoke,
        }

        if not ctx.dry_run:
            save_state(state)

        if baseline and not ctx.dry_run and not verify(baseline, stage.name):
            return 2

        if not ok:
            print(f"\nstopped at '{stage.name}'. Fix it and resume with "
                  f"--from {stage.name}", file=sys.stderr)
            return 1

        print(f"[{stage.name}] done in {elapsed / 60:.1f} min\n")

    if args.restore_2026 and not ctx.dry_run:
        print("restoring the 2026 Qwen3-VL server")
        teardown(ctx)
        start_qwen(ctx)
    elif (not args.keep_servers and not ctx.dry_run
            and _backend["current"] is not None):
        # Only tear down what this run brought up. A CPU-only invocation never
        # claimed the card, so whatever is on it belongs to someone else.
        teardown(ctx)

    print(f"all stages finished in {(time.monotonic() - started_all) / 60:.1f} "
          f"min")

    if ctx.smoke:
        print("\nSmoke run green. The real thing:\n"
              "    python VQA/run.eval.2025.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
