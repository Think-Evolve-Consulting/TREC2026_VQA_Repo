# Session Notes — 2026-08-13 — Video duration audit

Measuring how long the Task 1 videos actually are, plotting the distribution,
and pulling out the long tail. Probing and charting ran 12 August; the
long-video extract landed 13 August.

**Deadline context:** all Answer Generation (AG) runs are due **17 August 2026**
per `Welcome to the TREC2026 VQA Track_JULY2026.md`.

---

## Files created this session

| File | Purpose |
|---|---|
| `VQA/duration.report.py` | ffprobe every video, join to the query set, emit JSON + CSV |
| `VQA/duration_report_chart.py` | Plotly rendering, kept separate from probing |
| `duration.report.json` | Per-video durations plus corpus statistics (718 KB) |
| `duration.report.html` | Self-contained report page (4.7 MB, plotly.js inlined) |
| `videos.over.150s.csv` | The 65 videos longer than 150 s, with their Q_IDs |

The split between the two scripts exists so `--rebuild` can restyle the page
from cached probe data without re-reading 1,940 files.

---

## 1. Probing

```
python VQA/duration.report.py
```

**1,940 videos in 26.4 s at 8 workers (~74/s), zero failures.** ffprobe is
called once per file for `format=duration,size` plus per-stream
`codec_type,codec_name,width,height,duration`.

Design points:

- **Duration falls back.** Container duration first (`format.duration`), then
  the video stream's own. Matroska/WebM does not always carry a container
  duration, and a file with neither is recorded as an error rather than
  silently counted as zero. Nothing hit the fallback in this corpus, but it is
  the difference between a wrong total and a reported gap.
- **The query join follows `download_test_videos.py`'s naming convention** — a
  `.webm` is referenced with its extension, a YouTube pull by bare ID — so
  `Video_ID` resolves to a file with no lookup table.
- **Query counts ride along.** Each record carries `n_queries`, which makes
  query-weighted footage computable; see §4 for why that is the number that
  matters.

---

## 2. What the corpus looks like

| | n | median | max | total |
|---|---:|---:|---:|---:|
| Released clips (`.webm`) | 1,088 | 10.0 s | 389 s | 9.51 h |
| YouTube downloads (`.mp4`) | 852 | 15.1 s | 30.5 s | 3.85 h |
| **All** | **1,940** | **10.3 s** | **389 s** | **13.37 h** |

Three findings, all of which shaped the charts:

**The released clips are bimodal, not skewed.** 78.4% are ≤ 15 s, but 17.4%
exceed 60 s and the tail reaches 389 s. There is no smooth middle — **zero webm
clips fall between 15 s and 20 s**. Two populations share one label: short
extracts and near-full scenes.

**307 released clips are exactly 10.01 s** — 28% of the webm set sitting on a
single value, with another 55 within 0.05 s of it. This one spike is most of
what a histogram of the corpus shows.

**No YouTube download exceeds 30.5 s.** All 852 sit in a smooth, unimodal
5.2–30.49 s band. That is a suspiciously clean ceiling; worth confirming it
reflects the source dataset rather than a clip in `download_test_videos.py`,
because if it is ours, every one of those 852 videos is truncated.

Incidental facts from the same probe, both relevant to other pipelines:

- **5 videos carry no audio stream at all** — `6744.webm`, `6769.webm`,
  `7023.webm`, `7367.webm`, `7458.webm`. Anything ASR-driven has nothing to work
  with on these.
- **795 of 1,940 videos are portrait** (height > width), a large enough share
  that letterboxing or aspect handling in the VLM pass is not an edge case.
  Codecs split h264 852 / vp8 650 / vp9 438.

---

## 3. The report page

Two chart views, because one cannot honestly show a 1 s–389 s range:

1. **Histogram, 0–60 s in 2 s bins, stacked by source.** Covers 90% of the
   corpus at a readable bin width. The 189 videos above 60 s are stated on the
   chart as off-scale rather than crushed into a misleading final bin.
2. **ECDF on a log x-axis.** The whole range, no binning artifacts, both
   sources directly comparable.

Plus a hero figure, four stat tiles, and a summary-statistics table so every
plotted value is readable without the charts.

**Colors were validated, not eyeballed.** Categorical slots 1 (blue) and 2
(orange) from the reference palette, run through the validator for both
surfaces: worst adjacent CVD ΔE **24.7 light / 26.8 dark** against a floor of 8,
all six checks passing in both modes. The page follows the system theme; because
Plotly bakes colors into the figure, a small JS block restyles the traces on
`prefers-color-scheme` change rather than relying on the CSS variables that
drive the surrounding page.

### Three defects the render caught

Screenshotting the output — rather than trusting the code — found all three:

- **Legend order was reversed between the two charts.** Plotly flips the legend
  on stacked bars, so the histogram listed the sources in the opposite order to
  the ECDF. Fixed with `legend.traceorder: "normal"`; a reader should learn one
  order.
- **The ECDF annotation sat on top of both curves.** Moved into the empty
  quadrant right of where the orange curve tops out, with a leader line.
- **The annotations hardcoded their own findings.** The strings read "307
  released clips…" and "…30.5 s" as literals. Both are now derived from the
  data — the spike is found by counting repeated exact durations (and suppressed
  entirely below 5% of the corpus), the ceiling read from the per-source max.
  As written, the page would have kept asserting stale numbers the moment the
  remaining 868 videos landed.

---

## 4. The long tail — 65 videos over 150 s

```
python VQA/duration.report.py --rebuild --longer-than 150 --no-html
```

`--longer-than` was added to the existing script rather than written as a
one-off, so any threshold is repeatable against cached probe data. It writes
`videos.over.<N>s.csv` with duration, `mm:ss`, query count, and the Q_IDs.

**All 65 are released `.webm` clips** — no YouTube download comes close, since
that set stops at 30.5 s. They come from exactly ten films, with no singletons:

| Film | Clips | Film | Clips |
|---|---:|---|---:|
| Archipelago | 10 | little_rock | 7 |
| bonneville | 10 | heart_machine | 6 |
| losing_ground | 9 | Calloused_Hands | 3 |
| ChainedforLife | 7 | Memphis | 3 |
| Liberty_Kid | 7 | like_me | 3 |

Longest is `Archipelago-22.webm` at 389 s (6:29); the shortest over the line is
`losing_ground-7.webm` at 151 s.

### The disproportion, which runs both ways

| | 65 videos > 150 s | Top 15 by length |
|---|---:|---:|
| Share of videos | 3.35% | 0.77% |
| Share of queries | **6.34%** (155) | **1.56%** (38) |
| Share of raw footage | 28.79% | 9.11% |
| Share of query-weighted footage | **38.74%** | **13.00%** |
| Queries per video | 2.38 | 2.53 |

Corpus average is 1.26 queries per video, so long videos carry nearly **double**
the questions each — which is why their query-weighted share (38.74%) runs well
above their raw footage share (28.79%).

**The gap between 38.74% of the processing load and 6.34% of the score is the
number to act on.** A denser sampling strategy for long videos buys accuracy on
one query in sixteen while consuming two-fifths of the compute. Inversely,
capping or downsampling these 65 files leaves 94% of queries untouched.

Note the two footage totals are different measures: **13.37 h** is what is on
disk, **23.70 h** is query-weighted — a video counted once per query that asks
about it. The second is what predicts an inference pass, and is 77% larger.

---

## Command reference

```bash
# Full probe + report page
python VQA/duration.report.py

# Restyle the page from cached probe data, no ffprobe
python VQA/duration.report.py --rebuild

# Long-video extract at any threshold
python VQA/duration.report.py --rebuild --longer-than 150 --no-html
python VQA/duration.report.py --rebuild --longer-than 60 --no-html

# Smoke test
python VQA/duration.report.py --limit 40
```

---

## Environment verified

- `ffprobe` at `C:\ProgramData\chocolatey\bin`
- `plotly` 6.9.0 — installed this session; `requests` 2.34.2 already present
- Rendering checked headlessly in Chrome in both light and dark

---

## Next steps

1. **Confirm the 30.5 s ceiling on the YouTube pulls is upstream, not ours.**
   If `download_test_videos.py` is clipping, 852 videos are truncated and the
   answering pass is working from partial evidence.
2. **Decide the frame-sampling budget with these numbers in hand.** A fixed
   frames-per-video rate treats 389 s the same as the 10 s clips that dominate
   the set; a fixed frames-per-second rate spends 39% of the budget on 3% of the
   files.
3. Feed `videos.over.150s.csv` to whatever long-video handling gets built — it
   already carries the Q_IDs, so the affected queries are addressable directly.
4. The **5 videos with no audio** need a defined fallback in any ASR-dependent
   path (see `SESSION.2026-08-11.audio-asr-sensevoice.md` and
   `SESSION.2026-08-12.audio-asr-whisper.md`).
