#!/usr/bin/env python3
"""Run the two-model ensemble for TREC 2026 Task 1 end to end, unattended.

Implements ENSEMBLE_PLAN.md. Nine stages, one GPU, ~13 hours:

    deps -> download -> smoke:gemma4 -> smoke:qwen3vl
         -> answers:qwen3vl -> answers:gemma4 -> fuse -> expand -> validate

Output is ``fused.expanded.run.task1.json`` -- 2,443 queries, ten ranked
answers each, pooled from 60 candidates per query (10 prompts x k=3 x 2 models).

**Stage order is chosen to cost two server swaps, not four.** Both smoke tests
run before either long run, so an Instruct checkpoint that will not serve is
found in twenty minutes rather than five hours in; then Qwen3-VL is already
resident so its full run follows immediately, and Gemma 4 stays resident from
its own run through the expansion. Every swap is a teardown and a model load on
one 3090.

**The smoke gates are what make this safe to leave alone.** After each smoke run
the work cache is read back and checked for the three things that fail silently:
logprobs absent (the whole probability half of the ranking would be missing, and
the run would look perfectly healthy without it), the candidate pool collapsing
to a single call's worth, and -- for Qwen3-VL -- completion tokens that have not
dropped, which means the server is still on the ``-Thinking`` checkpoint and the
run would take 109 hours instead of 7. Any of those aborts before the long runs
start. See ``gate_smoke()``.

**Server lifecycle is imported, not re-implemented.** ``run.eval.2025.py`` paid
for that code with three multi-hour restarts -- process ownership on teardown,
session survival on launch, stdio handling -- and a second copy here would drift
from it. The functions come from there by importlib (the module name has dots,
so a plain import will not do it); only ``QWEN_SERVE`` is overridden, for the
Instruct checkpoint. See ``lifecycle()``.

**One interactive WSL2 session is held open for the whole run.** WSL2 tears the
distro down when its last session exits, taking the user systemd manager and
every vLLM server with it; ``loginctl enable-linger`` keeps the manager alive
while the distro runs but does not keep the distro alive. Until now that was a
human keeping a terminal open. ``keepalive()`` makes it the orchestrator's job.

Resumable: every stage records to ``ensemble.state.json`` and writes a marker
file, so a re-run skips what is done. Ctrl-C is safe -- the answering passes
cache per query and resume.

Usage:
    python VQA/run.ensemble.py --dry-run     # print the whole plan, run nothing
    python VQA/run.ensemble.py               # the real thing, ~13 h
    python VQA/run.ensemble.py --from fuse   # resume
    python VQA/run.ensemble.py --stages smoke:gemma4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
VQA = ROOT / "VQA"

STATE_FILE = ROOT / "ensemble.state.json"
LOG_DIR = ROOT / "logs.ensemble"

QUERIES = ROOT / "vqa.2026.testingDataset.Task1.json"
ROUTING = ROOT / "query.routing.task1.json"
VARIATIONS = ROOT / "query.variations.task1.json"

# New names throughout: the four validated runs already on disk are the
# fallback if this lands badly and must not be touched.
GEMMA_RUN = ROOT / "gemma4.ens.run.task1.json"
GEMMA_CACHE = ROOT / "answers.task1.gemma4.ens.cache.jsonl"
QWEN_RUN = ROOT / "qwen3vl.ens.run.task1.json"
QWEN_CACHE = ROOT / "answers.task1.qwen3vl.ens.cache.jsonl"
FUSED_RUN = ROOT / "fused.run.task1.json"
FUSED_CACHE = ROOT / "fused.task1.cache.jsonl"
EXPANDED_RUN = ROOT / "fused.expanded.run.task1.json"
EXPANDED_CACHE = ROOT / "answers.expansion.task1.fused.cache.jsonl"

SMOKE_GEMMA_RUN = ROOT / "smoke.ens.gemma4.json"
SMOKE_GEMMA_CACHE = ROOT / "smoke.ens.gemma4.cache.jsonl"
SMOKE_QWEN_RUN = ROOT / "smoke.ens.qwen3vl.json"
SMOKE_QWEN_CACHE = ROOT / "smoke.ens.qwen3vl.cache.jsonl"

WSL_DISTRO = "Ubuntu-24.04"
WSL_ROOT = "/mnt/e/TREC_2026"
WSL_PY = "/home/aakash/cosmos-reason2/.venv/bin/python"
WIN_PY = ROOT / ".venv" / "Scripts" / "python.exe"

QWEN_INSTRUCT = "Qwen/Qwen3-VL-8B-Instruct"
QWEN_INSTRUCT_DIR = "/home/aakash/models/Qwen3-VL-8B-Instruct"

# Same flags as the -Thinking server with two deliberate differences: the local
# checkpoint directory, and no --reasoning-parser. The Instruct checkpoint has
# no think block, and the parser was never doing anything anyway -- the 13 Aug
# run recorded reasoning_chars 0 on all 2,443 rows.
QWEN_INSTRUCT_SERVE = (
    "cd /home/aakash/cosmos-reason2 && "
    f"setsid nohup ./.venv/bin/vllm serve {QWEN_INSTRUCT_DIR} "
    "--host 0.0.0.0 --port 8009 "
    f"--served-model-name {QWEN_INSTRUCT} "
    f"--allowed-local-media-path {WSL_ROOT} "
    "--max-model-len 24576 --gpu-memory-utilization 0.92 --enforce-eager "
    "--media-io-kwargs '{\"video\": {\"num_frames\": -1}}' "
    "--skip-mm-profiling "
    "< /dev/null > /home/aakash/qwen3vl.instruct.log 2>&1 & disown; exit 0"
)

DOWNLOAD = (
    "cd /home/aakash/cosmos-reason2 && "
    f"./.venv/bin/huggingface-cli download {QWEN_INSTRUCT} "
    f"--local-dir {QWEN_INSTRUCT_DIR}"
)

SMOKE_N = 5

# Thresholds for the smoke gates. Deliberately loose -- these catch a stage that
# is broken, not one that is merely worse than hoped.
MIN_POOLED = 8
"""Distinct candidates a smoke query must pool from its 30. Below this the
ten-prompt ensemble is not actually happening."""

MAX_MEAN_TOKENS = 400
"""Mean completion tokens per *call* on Qwen3-VL. The -Thinking checkpoint ran
1,423; anything near that means the wrong model is being served and the full run
would take ~109 h instead of ~7.3 h."""


# ---------------------------------------------------------------------------
# Reused machinery
# ---------------------------------------------------------------------------

def lifecycle():
    """Import run.eval.2025.py for its GPU/server arbitration.

    Its module name has dots, so importlib is the only way in. Nothing runs at
    import: the file is constants, defs and a __main__ guard.

    The coupling is deliberate and narrow -- teardown(), the three start_*()
    functions, ensure_backend(), run_step() and the HTTP/VRAM waits. Every one
    of them encodes a failure this project has already paid for once (an Ollama
    child holding 8.9 GiB after its parent was killed; a vLLM server torn down
    by its launcher's own pipes; pkill matching its own shell). A second copy
    would drift from the fixed one.
    """
    path = VQA / "run.eval.2025.py"
    spec = importlib.util.spec_from_file_location("run_eval_2025", path)

    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")

    module = importlib.util.module_from_spec(spec)

    # Registered before exec: @dataclass resolves cls.__module__ through
    # sys.modules while the class body is being processed, and run.eval.2025
    # defines several. Without this the import dies inside dataclasses.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # The one substitution: serve Instruct, not Thinking. start_qwen() reads
    # this constant, so overriding it here reuses the whole start/wait path.
    module.QWEN_SERVE = QWEN_INSTRUCT_SERVE
    module.LOG_DIR = LOG_DIR

    return module


EV = None  # set in main(), the imported run.eval.2025 module


def shim(dry_run: bool):
    """The subset of run.eval.2025.Ctx that its lifecycle functions read."""
    return SimpleNamespace(dry_run=dry_run)


# ---------------------------------------------------------------------------
# Timestamped output
# ---------------------------------------------------------------------------

def stamp() -> str:
    """Local wall clock, to the second. A run spans 13 hours and a midnight."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def hms() -> str:
    return datetime.now().strftime("%H:%M:%S")


class Stamped:
    """Prefix every complete line written to a stream with the wall clock.

    Installed over sys.stdout/sys.stderr rather than stamping at each print,
    because the most diagnostically useful lines are not this module's: the
    backend swaps and the "still starting… 895s left" countdown come from the
    imported run.eval.2025 lifecycle, and on an unattended run the question is
    always *when* something stalled.

    Writes are buffered until a newline, so a partial write (a progress bar
    redrawing with \\r, which huggingface-cli does) does not get a stamp in the
    middle of a line.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self.partial = ""

    def write(self, text: str) -> int:
        self.partial += text

        while "\n" in self.partial:
            line, _, self.partial = self.partial.partition("\n")
            self.stream.write(f"{hms()} {line}\n")

        self.stream.flush()

        return len(text)

    def flush(self) -> None:
        self.stream.flush()

    def isatty(self) -> bool:
        return getattr(self.stream, "isatty", lambda: False)()

    def close(self) -> None:
        if self.partial:
            self.stream.write(f"{hms()} {self.partial}\n")
            self.partial = ""

        self.stream.flush()


def say(message: str = "", err: bool = False) -> None:
    """One orchestrator line. The stream wrapper adds the timestamp."""
    print(message, file=sys.stderr if err else sys.stdout)


def run_step(step, ctx) -> int:
    """Execute one step, tee'ing timestamped output to console and log file.

    A local copy of run.eval.2025.run_step rather than the imported one, and
    only because of the timestamps: on an unattended 13-hour run the useful
    question is almost always *when* something stalled, and an unstamped log
    cannot answer it. The process handling is otherwise identical -- line
    buffered, stderr folded into stdout, appended to a per-stage log.

    The stamps go into the log file as well as the console, since the console
    scrollback is the thing least likely to survive the run.
    """
    if step.host == "wsl":
        body = " ".join(shlex.quote(a) for a in step.argv)
        argv = EV.wsl_bash(f"cd {shlex.quote(WSL_ROOT)} && {body}")
    else:
        argv = step.argv

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{step.log}.log"

    say(f"  $ {step.display()}")

    if ctx.dry_run:
        return 0

    started = time.monotonic()

    with log_path.open("a", encoding="utf-8", errors="replace") as log:
        log.write(f"\n===== {stamp()} {step.display()}\n")
        log.flush()

        process = subprocess.Popen(
            argv, cwd=str(ROOT), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        )

        assert process.stdout is not None

        for line in process.stdout:
            # The console goes through the Stamped wrapper, which adds the
            # clock; the log file does not, so it stamps its own.
            sys.stdout.write("    | " + line)
            log.write(f"{hms()} {line}")
            log.flush()

        process.wait()

    elapsed = time.monotonic() - started
    say(f"  ({elapsed / 60:.1f} min, exit {process.returncode})")

    return process.returncode


# Where each server writes, so a failed start can show why instead of just
# timing out. start_qwen()/start_gemma() redirect to these.
SERVER_LOGS = {
    "vllm-qwen": "/home/aakash/qwen3vl.instruct.log",
    "vllm-gemma4": "/home/aakash/gemma4/eval.log",
}


def free_card(ctx) -> None:
    """Re-assert that nothing auto-starts onto the 3090 before a server does.

    ``cosmos-reason2`` is a systemd *user* unit, so it starts whenever the WSL2
    user manager starts -- which is every time the distro comes up, including
    the restart that any wsl.exe call triggers after an idle shutdown. It takes
    ~16 GB of the 24 and Qwen3-VL then cannot allocate at
    --gpu-memory-utilization 0.92: its engine core fails, and because stderr
    redirected to a file is block-buffered the server log is still *zero bytes*
    when it dies. The visible symptom is a fifteen-minute wait on a port that
    never opens.

    teardown() stops the unit, but stopping is not enough -- the distro can
    restart between the teardown and the launch and bring it straight back.
    Cosmos is dropped from this pipeline entirely, so the unit is disabled here
    as well, idempotently, on every server start.
    """
    if ctx.dry_run:
        return

    EV.quiet(EV.wsl_bash(
        "systemctl --user stop cosmos-reason2 2>/dev/null; "
        "systemctl --user disable cosmos-reason2 2>/dev/null; exit 0"
    ), timeout=120)


def show_server_log(backend: str, lines: int = 40) -> None:
    """Tail the server's own log after a failed start."""
    path = SERVER_LOGS.get(backend)

    if not path:
        return

    code, out = EV.quiet(EV.wsl_bash(
        f"echo '--- {path} ---'; "
        f"stat -c 'bytes: %s' {path} 2>/dev/null || echo 'no log file'; "
        f"tail -{lines} {path} 2>/dev/null"), timeout=120)

    print(f"\n  server log for {backend}:", file=sys.stderr)

    for line in (out or "").splitlines():
        print(f"    | {line}", file=sys.stderr)

    print("\n  (a zero-byte log means the process died before its first "
          "stderr flush -- almost always another process holding the card)",
          file=sys.stderr)


def bring_up(backend: str, ctx) -> bool:
    """ensure_backend, plus the two things it does not do by itself."""
    if backend not in SERVER_LOGS:
        return EV.ensure_backend(backend, ctx)

    # Re-checked per server start, not just once at the keepalive: the distro
    # can cycle mid-run, and a launch into a cold one dies silently.
    if not ctx.dry_run:
        wait_for_distro(120)

    free_card(ctx)

    if EV.ensure_backend(backend, ctx):
        return True

    used = EV.gpu_used_mib()
    print(f"  backend '{backend}' did not come up "
          f"(GPU holding {used} MiB)", file=sys.stderr)

    if not ctx.dry_run:
        holders = EV.gpu_holders()

        if holders:
            print(f"  processes on the card: {holders}", file=sys.stderr)

        show_server_log(backend)

    return False


def wait_for_distro(timeout: float = 180.0) -> bool:
    """Block until the distro is actually ready to host a vLLM server.

    A distro that answers `wsl.exe` is not the same as a distro that can start
    a CUDA process. Starting the keepalive *starts* the distro, and for the
    first several seconds systemd is still bringing units up and the GPU-PV
    device nodes are not there yet. A vLLM launched into that window dies
    immediately -- and because its stderr is block-buffered into a redirect,
    it dies without writing a single byte, so the symptom is a zero-length
    server log and a fifteen-minute wait on a port that never opens.

    Measured on this box: a launch 16 s after the distro came up produced
    exactly that; the identical command against a warm distro came up normally.

    Readiness here means the systemd user manager has finished starting and
    nvidia-smi can enumerate the card from inside WSL.
    """
    deadline = time.monotonic() + timeout
    last = ""

    while time.monotonic() < deadline:
        code, out = EV.quiet(EV.wsl_bash(
            "systemctl --user is-system-running 2>/dev/null; "
            "nvidia-smi -L 2>/dev/null | head -1; exit 0"), timeout=60)
        last = " ".join((out or "").split())

        # "degraded" is fine and usual here -- a failed unrelated unit must not
        # hold up a run. What matters is that startup has *finished*.
        started = any(s in last for s in ("running", "degraded"))

        if started and "GPU 0" in last:
            print(f"  distro ready: {last[:100]}")
            return True

        time.sleep(3)

    print(f"  WARNING: distro not confirmed ready after {timeout:.0f}s "
          f"(last: {last[:120]}). Server starts may fail silently.",
          file=sys.stderr)

    return False


def keepalive() -> subprocess.Popen | None:
    """Hold one WSL2 session open for the lifetime of this process.

    WSL2 shuts the distro down when its last session exits, which takes the
    systemd user manager and every vLLM server with it. Every previous long run
    here depended on a human leaving a terminal open; this is that human.
    """
    try:
        process = subprocess.Popen(
            ["wsl.exe", "-d", WSL_DISTRO, "--", "sleep", "infinity"],
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"  WSL2 keepalive session held (pid {process.pid})")
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"  WARNING: could not hold a WSL2 session open ({exc}). "
              f"If the distro tears down mid-run every server dies with it.",
              file=sys.stderr)
        return None

    # The keepalive is what starts the distro, so this wait belongs with it.
    wait_for_distro()

    return process


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

# -u on every child. Their stdout is a pipe, not a tty, so Python block-buffers
# it into 4-8 KB chunks: the orchestrator then receives a whole burst of
# progress lines at once and stamps them all with the arrival time. Measured on
# the first long run -- the console showed query 1448 while the work cache was
# already at 1542, a ~29 minute lag that looks exactly like a stall.
def win(*args) -> "EV.Step":
    return EV.Step(argv=[str(WIN_PY), "-u"] + [str(a) for a in args],
                   host="windows")


def wsl(*args) -> "EV.Step":
    return EV.Step(argv=[WSL_PY, "-u"] + [str(a) for a in args], host="wsl")


def w(path: Path) -> str:
    """A repo path as the WSL2 side sees it."""
    return EV.to_wsl(path)


def steps_download() -> list:
    step = EV.Step(argv=["bash", "-lc", DOWNLOAD], host="windows",
                   log="download")
    step.argv = EV.wsl_bash(DOWNLOAD)

    return [step]


def steps_smoke_gemma() -> list:
    step = win(VQA / "generate.answers.py",
               "--queries", QUERIES, "--routing", ROUTING,
               "--variations", VARIATIONS,
               "--out", SMOKE_GEMMA_RUN, "--cache", SMOKE_GEMMA_CACHE,
               "--limit", SMOKE_N, "--workers", 4)
    step.log = "smoke.gemma4"

    return [step]


def steps_smoke_qwen() -> list:
    step = wsl(f"{WSL_ROOT}/VQA/generate.answers.qwen3vl.py",
               "--queries", w(QUERIES), "--routing", w(ROUTING),
               "--variations", w(VARIATIONS),
               "--model", QWEN_INSTRUCT,
               "--out", w(SMOKE_QWEN_RUN), "--cache", w(SMOKE_QWEN_CACHE),
               "--limit", SMOKE_N, "--workers", 4)
    step.log = "smoke.qwen3vl"

    return [step]


def steps_answers_qwen(workers: int) -> list:
    step = wsl(f"{WSL_ROOT}/VQA/generate.answers.qwen3vl.py",
               "--queries", w(QUERIES), "--routing", w(ROUTING),
               "--variations", w(VARIATIONS),
               "--model", QWEN_INSTRUCT,
               "--out", w(QWEN_RUN), "--cache", w(QWEN_CACHE),
               "--workers", workers)
    step.log = "answers.qwen3vl"

    return [step]


def steps_answers_gemma(workers: int) -> list:
    step = win(VQA / "generate.answers.py",
               "--queries", QUERIES, "--routing", ROUTING,
               "--variations", VARIATIONS,
               "--out", GEMMA_RUN, "--cache", GEMMA_CACHE,
               "--workers", workers)
    step.log = "answers.gemma4"

    return [step]


def steps_fuse(weights: str | None) -> list:
    args = [VQA / "fuse.answers.py",
            "--queries", QUERIES,
            "--gemma4", GEMMA_CACHE, "--qwen3vl", QWEN_CACHE,
            "--out", FUSED_RUN, "--cache", FUSED_CACHE]

    if weights:
        args += ["--weights", weights]

    step = win(*args)
    step.log = "fuse"

    return [step]


def steps_expand(workers: int) -> list:
    # All four paths explicit. expand.answers.py defaults --queries and
    # --routing to the *2025* files, and 2025 Q_IDs are 1-2000 against 2026's
    # 1-2443, so a default here would answer 2,000 of the 2,443 queries against
    # unrelated questions -- and then validate cleanly.
    step = win(VQA / "expand.answers.py",
               "--run", FUSED_RUN,
               "--queries", QUERIES, "--routing", ROUTING,
               "--out", EXPANDED_RUN, "--cache", EXPANDED_CACHE,
               "--workers", workers)
    step.log = "expand"

    return [step]


def steps_validate() -> list:
    out = []

    for run in (FUSED_RUN, EXPANDED_RUN):
        step = win(VQA / "validate.ag.run.py", QUERIES, run,
                   "--warnings-as-errors",
                   "--report-json", run.with_suffix(".validation.json"))
        step.log = "validate"
        out.append(step)

    return out


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------

def read_cache(path: Path) -> list[dict]:
    records = []

    if not path.exists():
        return records

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()

            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return records


def gate_smoke(path: Path, label: str, check_tokens: bool) -> bool:
    """Read a smoke cache back and refuse to go on if it is quietly wrong.

    Everything checked here produces a *complete, valid-looking* run when it
    fails, which is why it is checked rather than eyeballed.
    """
    records = read_cache(path)
    print(f"  gate [{label}]: {len(records)} smoke records in {path.name}")

    if not records:
        print(f"  FAILED: no records -- the smoke run produced nothing",
              file=sys.stderr)
        return False

    ok = True

    scored = sum(1 for r in records for c in (r.get("candidates") or ())
                 if c.get("logprob_mean") is not None)
    total = sum(len(r.get("candidates") or ()) for r in records)

    print(f"    candidates with a logprob : {scored}/{total}")

    if not scored:
        print(f"  FAILED: no candidate carries a logprob. The server is not "
              f"returning them, so the probability half of the ranking would "
              f"be silently absent from a 13-hour run.", file=sys.stderr)
        ok = False

    pooled = [r.get("pooled_unique") or 0 for r in records]
    mean_pooled = sum(pooled) / len(pooled)
    print(f"    distinct candidates pooled: mean {mean_pooled:.1f} "
          f"(min {min(pooled)}, want >= {MIN_POOLED})")

    if mean_pooled < MIN_POOLED:
        print(f"  FAILED: the ten-prompt ensemble is not happening -- "
              f"{mean_pooled:.1f} distinct candidates is one call's worth, "
              f"not ten.", file=sys.stderr)
        ok = False

    replies = [r.get("n_replies") or 0 for r in records]

    if replies:
        print(f"    prompt calls per query    : mean "
              f"{sum(replies) / len(replies):.1f} (want 10)")

    failures = sum(r.get("n_prompt_failures") or 0 for r in records)

    if failures:
        print(f"    WARNING: {failures} prompt calls failed outright")

    if check_tokens:
        per_call = []

        for record in records:
            calls = record.get("n_replies") or 0
            tokens = record.get("completion_tokens") or 0

            if calls:
                per_call.append(tokens / calls)

        if per_call:
            mean_tokens = sum(per_call) / len(per_call)
            print(f"    completion tokens per call: mean {mean_tokens:.0f} "
                  f"(want < {MAX_MEAN_TOKENS}; -Thinking ran 1423)")

            if mean_tokens >= MAX_MEAN_TOKENS:
                print(f"  FAILED: the served checkpoint is still generating "
                      f"like -Thinking. At this rate the full run is ~109 h, "
                      f"not ~7.3 h. Check that port 8009 serves "
                      f"{QWEN_INSTRUCT}.", file=sys.stderr)
                ok = False

    fallback = sum(r.get("fallback") or 0 for r in records)
    print(f"    fallback slots            : {fallback}/{len(records) * 10}")

    return ok


def stage_deps(ctx) -> bool:
    """Everything that would otherwise fail hours in."""
    ok = True

    for label, path in (("queries", QUERIES), ("routing", ROUTING),
                        ("variations", VARIATIONS)):
        mark = "ok " if path.exists() else "MISSING"
        print(f"  {mark} {label:<11} {path.name}")
        ok = ok and path.exists()

    if not WIN_PY.exists():
        print(f"  MISSING windows python at {WIN_PY}", file=sys.stderr)
        ok = False

    code, out = EV.quiet(["wsl.exe", "-d", WSL_DISTRO, "--", "bash", "-lc",
                          f"test -x {WSL_PY} && echo yes"], timeout=120)

    if "yes" not in out:
        print(f"  MISSING wsl python at {WSL_PY} ({out.strip()[:120]})",
              file=sys.stderr)
        ok = False
    else:
        print(f"  ok  wsl python  {WSL_PY}")

    # Cosmos is dropped from this pipeline, and an enabled unit auto-starts
    # with the WSL2 user manager onto a card that has room for exactly one
    # model. Asserted here as well as before each server start, so the state is
    # visible at the top of the log rather than inferred from a failure.
    free_card(ctx)
    code, out = EV.quiet(EV.wsl_bash(
        "systemctl --user is-enabled cosmos-reason2 2>/dev/null; "
        "systemctl --user is-active cosmos-reason2 2>/dev/null; exit 0"),
        timeout=120)
    print(f"  cosmos-reason2: {' '.join((out or '').split()) or 'absent'}")

    used = EV.gpu_used_mib()
    print(f"  gpu in use: {used} MiB" if used is not None
          else "  WARNING: nvidia-smi did not answer")

    if used and used > 2048:
        print(f"  WARNING: {used} MiB already held. {EV.gpu_holders()}")

    # These four runs are the fallback if the ensemble lands badly. Losing them
    # to a mistyped --out is not recoverable in the time available.
    for name in ("gemma4.run.task1.json", "gemma4.expanded.run.task1.json",
                 "qwen3vl.run.task1.json", "qwen3vl.expanded.run.task1.json"):
        if not (ROOT / name).exists():
            print(f"  WARNING: fallback submission {name} is missing")

    return ok


# ---------------------------------------------------------------------------
# Stage graph
# ---------------------------------------------------------------------------

@dataclass
class Stage:
    name: str
    backend: str
    build: object = None       # (args) -> list[Step]
    body: object = None        # (args) -> bool
    gate: object = None        # (args) -> bool, run after the steps
    marker: Path | None = None
    description: str = ""


def build_stages(args) -> list[Stage]:
    return [
        Stage("deps", "any", body=stage_deps,
              description="inputs, interpreters, GPU, fallback runs"),
        Stage("download", "any", build=lambda a: steps_download(),
              description=f"fetch {QWEN_INSTRUCT} (~16 GB, no GPU)"),
        Stage("smoke:gemma4", "vllm-gemma4",
              build=lambda a: steps_smoke_gemma(),
              gate=lambda a: gate_smoke(SMOKE_GEMMA_CACHE, "gemma4", False),
              marker=SMOKE_GEMMA_RUN,
              description=f"{SMOKE_N} queries, then check logprobs and pooling"),
        Stage("smoke:qwen3vl", "vllm-qwen",
              build=lambda a: steps_smoke_qwen(),
              gate=lambda a: gate_smoke(SMOKE_QWEN_CACHE, "qwen3vl", True),
              marker=SMOKE_QWEN_RUN,
              description=f"{SMOKE_N} queries, then also check the token "
                          f"collapse"),
        Stage("answers:qwen3vl", "vllm-qwen",
              build=lambda a: steps_answers_qwen(a.workers_qwen),
              marker=QWEN_RUN,
              description="2,443 x 10 prompts x k=3, ~7.3 h"),
        Stage("answers:gemma4", "vllm-gemma4",
              build=lambda a: steps_answers_gemma(a.workers_gemma),
              marker=GEMMA_RUN,
              description="2,443 x 10 prompts x k=3, ~4-5 h"),
        Stage("fuse", "any", build=lambda a: steps_fuse(a.weights),
              marker=FUSED_RUN,
              description="pool 60 candidates per query -> ten ranked"),
        Stage("expand", "vllm-gemma4",
              build=lambda a: steps_expand(a.workers_gemma),
              marker=EXPANDED_RUN,
              description="length-target rewrite of the fused run"),
        Stage("validate", "any", build=lambda a: steps_validate(),
              description="run-format validation of both fused runs"),
    ]


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    return {"stages": {}}


def save_state(state: dict) -> None:
    STATE_FILE.write_text(json.dumps(state, indent=4) + "\n",
                          encoding="utf-8")


def select(args, stages: list[Stage]) -> list[Stage]:
    names = [s.name for s in stages]

    if args.stages:
        unknown = [n for n in args.stages if n not in names]

        if unknown:
            raise SystemExit(f"unknown stage(s): {', '.join(unknown)}\n"
                             f"known: {', '.join(names)}")

        return [s for s in stages if s.name in args.stages]

    start = names.index(args.start) if args.start else 0
    stop = names.index(args.stop) + 1 if args.stop else len(names)

    return stages[start:stop]


def parse_args(argv=None) -> argparse.Namespace:
    stage_names = [s.name for s in build_stages(SimpleNamespace(
        workers_qwen=4, workers_gemma=8, weights=None))]

    ap = argparse.ArgumentParser(
        description="Run the two-model ensemble for TREC 2026 Task 1 "
                    "end to end.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages: " + " -> ".join(stage_names),
    )

    ap.add_argument("--dry-run", action="store_true",
                    help="print every command and gate, run nothing")
    ap.add_argument("--from", dest="start", metavar="STAGE",
                    help="resume from this stage")
    ap.add_argument("--to", dest="stop", metavar="STAGE",
                    help="stop after this stage")
    ap.add_argument("--stages", nargs="+", metavar="STAGE",
                    help="run exactly these stages")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages whose marker file already exists")
    ap.add_argument("--workers-qwen", type=int, default=4)
    ap.add_argument("--workers-gemma", type=int, default=8)
    ap.add_argument("--weights", help="ranking weights for fuse.answers.py")
    ap.add_argument("--no-keepalive", action="store_true",
                    help="do not hold a WSL2 session open (only if you are "
                         "keeping one open yourself)")
    ap.add_argument("--keep-servers", action="store_true",
                    help="leave the last server running at the end")

    return ap.parse_args(argv)


def main(argv=None) -> int:
    global EV

    args = parse_args(argv)

    if os.name != "nt":
        print("This orchestrator runs on Windows: it drives the WSL2 stages "
              "through wsl.exe and the Gemma 4 client natively.",
              file=sys.stderr)
        return 1

    # Installed before anything prints, and over stderr too, so the imported
    # lifecycle's output is stamped as well as this module's.
    sys.stdout = Stamped(sys.stdout)
    sys.stderr = Stamped(sys.stderr)

    EV = lifecycle()
    ctx = shim(args.dry_run)

    stages = select(args, build_stages(args))
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    say(f"TREC 2026 Task 1 — two-model ensemble")
    say(f"  target:  {EXPANDED_RUN.name} (2,443 queries x 10 answers)")
    say(f"  stages:  {' -> '.join(s.name for s in stages)}")
    say(f"  logs:    {LOG_DIR}")
    say(f"  state:   {STATE_FILE.name}")
    say()

    held = None if (args.dry_run or args.no_keepalive) else keepalive()
    state = load_state()
    started_all = time.monotonic()

    try:
        for stage in stages:
            if stage.marker and stage.marker.exists() and not args.force:
                say(f"[{stage.name}] already done "
                    f"({stage.marker.name}) — skipping")

                # A gated stage is re-checked even when skipped. The marker is
                # written by the step, not by the gate, so a smoke run that
                # produced output and then failed its gate would otherwise be
                # skipped on resume -- reading as "already done" precisely when
                # it is known not to be.
                if stage.gate and not args.dry_run and not stage.gate(args):
                    print(f"\n'{stage.name}' is on disk but does not pass its "
                          f"gate. Delete {stage.marker.name} and re-run, or "
                          f"fix the cause.", file=sys.stderr)
                    return 1

                print()
                continue

            say(f"[{stage.name}] {stage.description}")
            started = time.monotonic()

            if not bring_up(stage.backend, ctx):
                return 1

            ok = True

            if stage.body:
                ok = stage.body(ctx)

            if ok and stage.build:
                for step in stage.build(args):
                    code = run_step(step, ctx)

                    if code not in step.ok_returncodes:
                        say(f"  FAILED (exit {code})", err=True)
                        ok = False
                        break

            if ok and stage.gate and not args.dry_run:
                ok = stage.gate(args)

            elapsed = time.monotonic() - started
            state["stages"][stage.name] = {
                "ok": ok,
                "finished": datetime.now(timezone.utc)
                .isoformat(timespec="seconds"),
                "minutes": round(elapsed / 60, 2),
            }

            if not args.dry_run:
                save_state(state)

            if not ok:
                say(f"stopped at '{stage.name}' after "
                    f"{elapsed / 60:.1f} min. Fix it and resume with "
                    f"--from {stage.name}", err=True)
                return 1

            say(f"[{stage.name}] done in {elapsed / 60:.1f} min")
            say()

    except KeyboardInterrupt:
        print("\ninterrupted -- the answering passes cache per query, so "
              "re-running resumes where this stopped", file=sys.stderr)
        return 130
    finally:
        if held is not None:
            held.terminate()

        if not args.dry_run and not args.keep_servers:
            EV.teardown(ctx)

        for stream in (sys.stdout, sys.stderr):
            if isinstance(stream, Stamped):
                stream.close()

    total = (time.monotonic() - started_all) / 3600
    say(f"finished in {total:.2f} h")
    say(f"  submission candidate: {EXPANDED_RUN}")
    say(f"  unexpanded:           {FUSED_RUN}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
