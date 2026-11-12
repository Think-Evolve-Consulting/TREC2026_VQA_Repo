# Session Notes — 2026-08-11 — VQA 2026 Task 1 run pipeline

Working session covering three things: a format-valid dummy run file, a local
video coverage audit, and a downloader for the missing test videos.

**Deadline context:** all Answer Generation (AG) runs are due **17 August 2026**
per `Welcome to the TREC2026 VQA Track_JULY2026.md`.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/generate.dummy.run.py` | Generates a random placeholder Task 1 run file |
| `dummy.run.task1.json` | The generated run — 2,443 entries, 4 MB, passes the validator |
| `VQA/coverage.report.py` | Audits which query videos exist on disk |
| `coverage.report.json` | Machine-readable coverage output |
| `download_test_videos.py` | Downloads the missing YouTube videos with correct naming |

Pre-existing files referenced: `VQA/validate.ag.run.py`, `batch_download.py`,
`retry_downloads.py`, `vqa.2026.testingDataset.Task1.json`.

---

## 1. Dummy run file

Purpose: exercise the submission format end-to-end before any real model output
exists, so format bugs surface now rather than on 17 August.

```bash
python VQA/generate.dummy.run.py vqa.2026.testingDataset.Task1.json dummy.run.task1.json
python VQA/validate.ag.run.py vqa.2026.testingDataset.Task1.json dummy.run.task1.json --warnings-as-errors
```

Result:

```
VALIDATION PASSED
Expected queries:    2443
Submitted entries:   2443
Errors:              0
Warnings:            0
```

### Format constraints enforced by `validate.ag.run.py`

These are the rules the generator satisfies — the same ones the real run must:

- Top-level JSON array; each entry has **exactly** `Q_ID`, `Video_ID`, `Answers`.
  Extra fields are errors unless `--allow-extra-fields` is passed.
- Each answer object has **exactly** `Rank`, `Answer`, `Time`.
- Exactly **10** answers per query (`--answers-per-query` to change).
- `Rank` is an integer 1–10, each used once. Out-of-order ranks are a *warning*,
  missing/duplicate ranks are *errors*.
- `Answer` is a non-empty string. Duplicates within one query are an error,
  compared after whitespace collapsing and case folding.
- `Time` is a finite number ≥ 0.
- Every `Q_ID` from the query file must appear exactly once, and its `Video_ID`
  must match the released value verbatim.
- Strict JSON parsing: duplicate keys, `NaN`, and `Infinity` are all rejected.

The generator is reproducible via `--seed` (default 2026).

> **Note from `VQA/README.md`:** validator error paths are 0-indexed while
> `Q_ID` starts at 1, so `submission[41]` means `Q_ID` 42.

---

## 2. Local video coverage

```bash
python VQA/coverage.report.py vqa.2026.testingDataset.Task1.json --report-json coverage.report.json
```

| | Referenced | Present | Missing |
|---|---|---|---|
| `.webm` clips (`videos_TEST_VQA/videos/`) | 1,088 | **1,088** | 0 |
| YouTube IDs (checked in `videos/`) | 868 | **0** | 868 |
| Distinct `Video_ID`s | 1,956 | 1,088 | 868 |

**1,575 / 2,443 queries (64.5%) are answerable from local media. 868 are blocked**
pending downloads; their Q_IDs are in `blocked_q_ids` in `coverage.report.json`.

Findings worth remembering:

- The webm folder is an exact 1:1 match with the webm portion of the query set —
  nothing missing, nothing unused.
- **The 473 mp4s in `videos/` are entirely unused by the 2026 test set.** Per
  `download_summary.json` they came from `vqa.2025.trainingData` — last year's
  training downloads. Zero overlap with this year's queries.
- **None of the 19 IDs in `Videos Removed from the platform.txt` appear in the
  2026 test set**, so that known-dead list does not reduce the 868.
- No case-only filename collisions (relevant because YouTube IDs are
  case-sensitive but NTFS is not).
- All 868 YouTube IDs are clean 11-character IDs — no URLs, no odd formats.
- Videos are reused across queries: 1,956 distinct videos serve 2,443 questions,
  the busiest carrying 5 questions each.

---

## 3. Downloader for the missing videos

`download_test_videos.py` reads the query JSON directly (rather than a folder of
per-video JSONs like `batch_download.py` does), dedupes the 868 YouTube IDs, and
fetches them.

### Naming convention — the important part

Videos are written as `<Video_ID>.mp4` into `videos_TEST_VQA/videos/`, beside the
released `.webm` clips. The YouTube ID is the file stem **verbatim** — no
sanitising, no title-based names — so resolving a query to a file stays a
one-liner with no lookup table:

```python
name = vid if vid.endswith(".webm") else vid + ".mp4"
path = Path("videos_TEST_VQA/videos") / name
```

One directory therefore serves all 2,443 queries. Metadata goes to
`videos_TEST_VQA/metadata/<Video_ID>.json` (same layout `batch_download.py`
produces) with an added `task1_queries` field listing the Q_IDs and questions
asked about that video.

### What it reuses vs. adds

Reuses `download_one` from `batch_download.py` unchanged — yt_dlp fetch, mp4
remux, AV1/VP9/VP8 → H.264 re-encode, metadata write — plus the
exponential-backoff loop from `retry_downloads.py`. Added on top:

- **Resumable.** Any `Video_ID` already on disk is skipped; the summary is
  checkpointed after every video. Ctrl-C is trapped mid-download and writes a
  partial summary.
- **Permanent-failure detection** — "video unavailable", "private video",
  terminated account, etc. skip the retry loop instead of burning three attempts
  plus backoff on a dead video.
- **Fatal-config detection** — cookie/browser errors abort the entire run rather
  than repeating 868 times (see the cookie issue below).
- **`--cookies-from-browser`**, injected via a `YoutubeDL` subclass since
  `download_one` only accepts a cookies.txt path.
- **`--temp-dir`** — see "Why `--temp-dir`" below.
- `--limit`, `--only`, `--retry-failed` for smoke tests and re-runs;
  `--sleep` (default 2 s, jittered ±25%) to stay polite across 868 fetches.

Summary is written to `videos_TEST_VQA/download_summary.task1.json`.

### Why `--temp-dir`

`download_one` stages into `tempfile.TemporaryDirectory()`, which resolves to
`C:\Users\ADMIN\AppData\Local\Temp`. The final move is `os.replace`, which is an
instant rename within a drive but raises `OSError` across C: → E:, falling back
to `shutil.copy2` — a full second write of every file. Pointing `--temp-dir` at
something on E: (e.g. `videos_TEST_VQA\.tmp`) restores the rename and removes
that copy. Optional; correctness is unaffected either way.

---

## Open issue: Chrome cookies

`--cookies-from-browser chrome` fails with:

```
ERROR: Could not copy Chrome cookie database.
https://github.com/yt-dlp/yt-dlp/issues/7271
```

**Cause:** Chrome was running (55 processes) and holds an exclusive lock on
`Network\Cookies`, which yt-dlp must copy before reading.

Neither existing cookie file is usable: `cookies.txt` is a single token string,
not a Netscape cookie jar; `cookies.json` is a Cookie-Editor export whose
payload is an encoded blob.

Options, in order of preference:

1. **Skip cookies entirely.** Most of these videos are public and download fine
   anonymously from a residential IP. Test with `--limit 3` first.
2. **Quit Chrome completely** (including the background tray process) and retry.
   Caveat: Chrome 127+ uses App-Bound Encryption, which yt-dlp cannot decrypt
   even with Chrome closed — this may turn the copy error into a decrypt error.
3. **Use Firefox** — no lock, no App-Bound Encryption. The reliable browser path.
4. **Export a real Netscape `cookies.txt`** via a browser extension and pass
   `--cookies`.

If cookies are needed, use a throwaway account: 868 sequential authenticated
downloads is exactly the pattern YouTube flags.

---

## Environment verified

- `yt_dlp` 2025.11.12
- `ffmpeg` / `ffprobe` present at `C:\ProgramData\chocolatey\bin`
- Python on Windows 11; repo root `E:\TREC_2026` (media on E:, system temp on C:)

---

## Command reference

```bash
# Regenerate a dummy run with a different seed
python VQA/generate.dummy.run.py vqa.2026.testingDataset.Task1.json out.json --seed 123

# Validate any run (strict)
python VQA/validate.ag.run.py vqa.2026.testingDataset.Task1.json out.json --warnings-as-errors

# Re-check coverage after downloading
python VQA/coverage.report.py vqa.2026.testingDataset.Task1.json --report-json coverage.report.json

# Smoke-test the downloader
python download_test_videos.py --limit 3 --temp-dir videos_TEST_VQA\.tmp

# Full download run
python download_test_videos.py --temp-dir videos_TEST_VQA\.tmp --sleep 2

# Retry only what failed
python download_test_videos.py --retry-failed videos_TEST_VQA/download_summary.task1.json
```

---

## Next steps

1. Resolve the cookie situation (or confirm anonymous downloads work) and run the
   full download.
2. Re-run `coverage.report.py` to confirm coverage and see what remains
   permanently unavailable.
3. Decide how to handle queries whose video can never be fetched — the validator
   requires **all 2,443** Q_IDs present with 10 answers each, so those still need
   placeholder answers in the final submission.
4. Swap the dummy answer generator for real model output, keeping the same
   serialisation and re-validating before submission.
5. Confirm the data agreement form is submitted — runs are not scored without it.
