#!/usr/bin/env python3
"""Plotly rendering for duration.report.py.

Kept separate from the probing so the page can be restyled with --rebuild
without touching ffprobe. Two views, because one chart cannot honestly show
this distribution: 90% of the corpus sits under 60 s while the tail reaches
389 s, so the histogram covers the bulk at a readable bin width and the ECDF
carries the full range on a log axis.

Colors are the reference categorical slots 1 (blue) and 2 (orange), validated
for both surfaces -- worst adjacent CVD deltaE 24.7 light / 26.8 dark against
a floor of 8.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import plotly.graph_objects as go

BIN_WIDTH = 2
BIN_MAX = 60

# Slot 1 / slot 2 of the reference categorical theme, plus the chrome tokens.
THEME = {
    "light": {
        "surface": "#fcfcfb",
        "plane": "#f9f9f7",
        "primary": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "border": "rgba(11,11,11,0.10)",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "plane": "#0d0d0d",
        "primary": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "border": "rgba(255,255,255,0.10)",
        "series": ["#3987e5", "#d95926"],
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def sources(report: dict) -> list[str]:
    """Source labels in a fixed order, so a hue never moves between slots."""
    return list(report["by_source"].keys())


def durations_by_source(report: dict) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {s: [] for s in sources(report)}

    for record in report["videos"]:
        if "duration_sec" in record:
            grouped[record["source"]].append(record["duration_sec"])

    return {s: sorted(v) for s, v in grouped.items()}


def short(source: str) -> str:
    """'Released clips (.webm)' -> 'Released clips'."""
    return source.split(" (")[0]


def modal_duration(report: dict) -> tuple[str, float, int] | None:
    """The single most repeated exact duration, if it is worth pointing at.

    Derived rather than hardcoded: the spike moves as videos are added, and an
    annotation stating a stale count would be worse than none at all.
    """
    counts = Counter(
        (r["source"], round(r["duration_sec"], 2))
        for r in report["videos"] if "duration_sec" in r
    )

    if not counts:
        return None

    (source, value), n = counts.most_common(1)[0]
    total = report["videos_with_duration"]

    # Below ~5% of the corpus it is a coincidence, not a feature of the data.
    if total and n / total < 0.05:
        return None

    return source, value, n


def base_layout(theme: dict, height: int) -> dict:
    return {
        "height": height,
        "paper_bgcolor": theme["surface"],
        "plot_bgcolor": theme["surface"],
        "font": {"family": FONT, "size": 13, "color": theme["secondary"]},
        "margin": {"l": 70, "r": 30, "t": 20, "b": 60},
        "hoverlabel": {"font": {"family": FONT, "size": 13},
                       "bordercolor": theme["axis"]},
        "legend": {
            "orientation": "h",
            "yanchor": "bottom", "y": 1.02,
            "xanchor": "left", "x": 0,
            "font": {"color": theme["secondary"]},
            # Plotly reverses the legend on stacked bars; pinning it keeps the
            # two charts in the same order, so a reader learns one order.
            "traceorder": "normal",
        },
        "bargap": 0.08,
    }


def axis_style(theme: dict, title: str) -> dict:
    return {
        "title": {"text": title, "font": {"color": theme["secondary"],
                                          "size": 13}},
        "gridcolor": theme["grid"],
        "linecolor": theme["axis"],
        "zerolinecolor": theme["axis"],
        "tickfont": {"color": theme["muted"], "size": 12},
        "showline": True,
        "ticks": "outside",
        "tickcolor": theme["axis"],
    }


def build_histogram(report: dict, theme: dict) -> go.Figure:
    """Stacked histogram of the bulk of the corpus, 0-60 s in 2 s bins."""
    grouped = durations_by_source(report)
    edges = list(range(0, BIN_MAX + BIN_WIDTH, BIN_WIDTH))
    centres = [e + BIN_WIDTH / 2 for e in edges[:-1]]

    figure = go.Figure()
    overflow = 0
    stacked = [0] * (len(edges) - 1)

    for slot, source in enumerate(sources(report)):
        values = grouped[source]
        counts = [0] * (len(edges) - 1)

        for value in values:
            if value >= BIN_MAX:
                overflow += 1
                continue

            counts[int(value // BIN_WIDTH)] += 1

        stacked = [a + b for a, b in zip(stacked, counts)]

        figure.add_bar(
            x=centres,
            y=counts,
            name=source,
            marker={
                "color": theme["series"][slot],
                # The separator is a gap in the surface colour, not a border:
                # it reads as air between stacked segments and neighbours.
                "line": {"color": theme["surface"], "width": 1},
            },
            customdata=[[e, e + BIN_WIDTH] for e in edges[:-1]],
            hovertemplate=(
                "%{customdata[0]:.0f}-%{customdata[1]:.0f} s<br>"
                "%{y} videos<extra>" + source + "</extra>"
            ),
        )

    layout = base_layout(theme, 420)
    layout["barmode"] = "stack"
    layout["xaxis"] = {
        **axis_style(theme, "Clip duration (seconds)"),
        "range": [0, BIN_MAX],
        "dtick": 10,
    }
    layout["yaxis"] = {**axis_style(theme, "Videos"), "rangemode": "tozero"}

    layout["annotations"] = [
        {
            "x": 1, "y": 1.06, "xref": "paper", "yref": "paper",
            "text": f"{overflow} videos longer than {BIN_MAX} s are off-scale "
                    "— see the cumulative view",
            "showarrow": False, "xanchor": "right",
            "font": {"color": theme["muted"], "size": 12},
        },
    ]

    # One direct label, on the single feature that explains the whole shape.
    spike = modal_duration(report)

    if spike and spike[1] < BIN_MAX:
        source, value, n = spike
        centre = (value // BIN_WIDTH) * BIN_WIDTH + BIN_WIDTH / 2

        layout["annotations"].insert(0, {
            "x": centre + 0.2, "y": stacked[int(value // BIN_WIDTH)],
            "text": f"{n} {short(source).lower()} are<br>"
                    f"exactly {value:g} s long",
            "showarrow": True, "arrowhead": 0, "arrowwidth": 1,
            "arrowcolor": theme["axis"], "ax": 74, "ay": -30,
            "font": {"color": theme["secondary"], "size": 12},
            "align": "left",
        })

    layout["showlegend"] = len(figure.data) > 1
    figure.update_layout(**layout)
    return figure


def build_ecdf(report: dict, theme: dict) -> go.Figure:
    """Cumulative view on a log axis — the whole range, no binning."""
    grouped = durations_by_source(report)
    figure = go.Figure()

    for slot, source in enumerate(sources(report)):
        values = grouped[source]
        total = len(values)

        figure.add_scatter(
            x=values,
            y=[100 * (i + 1) / total for i in range(total)],
            name=source,
            mode="lines",
            line={"color": theme["series"][slot], "width": 2,
                  "shape": "hv"},
            hovertemplate=("%{x:.1f} s<br>%{y:.1f}% at or below"
                           "<extra>" + source + "</extra>"),
        )

    ticks = [1, 2, 5, 10, 15, 30, 60, 120, 300]

    layout = base_layout(theme, 380)
    layout["xaxis"] = {
        **axis_style(theme, "Clip duration (seconds, log scale)"),
        "type": "log",
        "tickvals": ticks,
        "ticktext": [str(t) for t in ticks],
        "range": [0, 2.62],
    }
    layout["yaxis"] = {
        **axis_style(theme, "Share of videos at or below"),
        "range": [0, 100],
        "dtick": 25,
        "ticksuffix": "%",
    }
    # Point at whichever source stops first — the ceiling is the finding here.
    capped = min(report["by_source"].items(), key=lambda kv: kv[1]["max_sec"])
    ceiling = capped[1]["max_sec"]

    layout["annotations"] = [
        {
            # Log axis: annotation x is log10(seconds).
            "x": math.log10(ceiling), "y": 100,
            # Phrased for a plural label, so any source name reads correctly.
            "text": f"{short(capped[0])}<br>stop at {ceiling:g} s",
            "showarrow": True, "arrowhead": 0, "arrowwidth": 1,
            # Into the empty quadrant right of where that curve tops out;
            # anything closer overlaps both lines.
            "arrowcolor": theme["axis"], "ax": 92, "ay": 120,
            "font": {"color": theme["secondary"], "size": 12},
            "align": "left",
        }
    ] if len(report["by_source"]) > 1 else []

    layout["showlegend"] = len(figure.data) > 1
    figure.update_layout(**layout)
    return figure


def stat_tiles(report: dict) -> str:
    overall = report["overall"]

    tiles = [
        ("Videos measured", f"{report['videos_with_duration']:,}"),
        ("Median duration", f"{overall['median_sec']:.1f} s"),
        ("Longest clip", f"{overall['max_sec']:.0f} s"),
        ("Query-weighted footage", f"{report['query_weighted_hours']:.1f} h"),
    ]

    cells = "".join(
        f'<div class="tile"><div class="tile-label">{label}</div>'
        f'<div class="tile-value">{value}</div></div>'
        for label, value in tiles
    )

    return (
        '<div class="hero-row">'
        '<div class="hero"><div class="tile-label">Total footage on disk</div>'
        f'<div class="hero-value">{report["overall"]["total_hours"]:.1f} h</div>'
        "</div>"
        f'<div class="tiles">{cells}</div>'
        "</div>"
    )


def stats_table(report: dict) -> str:
    """The table view — every plotted value readable without the chart."""
    columns = [
        ("count", "n", "{:,}"),
        ("min_sec", "min", "{:.1f} s"),
        ("p25_sec", "p25", "{:.1f} s"),
        ("median_sec", "median", "{:.1f} s"),
        ("p75_sec", "p75", "{:.1f} s"),
        ("p95_sec", "p95", "{:.1f} s"),
        ("max_sec", "max", "{:.1f} s"),
        ("mean_sec", "mean", "{:.1f} s"),
        ("total_hours", "total", "{:.2f} h"),
    ]

    head = "".join(f"<th>{label}</th>" for _, label, _ in columns)
    rows = []

    for name, block in list(report["by_source"].items()) + [
        ("All videos", report["overall"])
    ]:
        cells = "".join(f"<td>{fmt.format(block[key])}</td>"
                        for key, _, fmt in columns)
        rows.append(f"<tr><th scope='row'>{name}</th>{cells}</tr>")

    return (
        f'<table><thead><tr><th scope="col">Source</th>{head}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def write_html(report: dict, path: Path) -> None:
    light = THEME["light"]
    histogram = build_histogram(report, light)
    ecdf = build_ecdf(report, light)

    hist_div = histogram.to_html(full_html=False, include_plotlyjs="inline",
                                 div_id="hist", config={"displaylogo": False,
                                                        "responsive": True})
    ecdf_div = ecdf.to_html(full_html=False, include_plotlyjs=False,
                            div_id="ecdf", config={"displaylogo": False,
                                                   "responsive": True})

    page = HTML_TEMPLATE.format(
        generated=report["generated"],
        n_videos=f"{report['videos_with_duration']:,}",
        n_queries=f"{report['queries_total']:,}",
        tiles=stat_tiles(report),
        histogram=hist_div,
        ecdf=ecdf_div,
        table=stats_table(report),
        themes=json.dumps(THEME),
    )

    path.write_text(page, encoding="utf-8")


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>TREC 2026 VQA Task 1 — video duration distribution</title>
<style>
  :root {{
    color-scheme: light;
    --surface: #fcfcfb; --plane: #f9f9f7;
    --primary: #0b0b0b; --secondary: #52514e; --muted: #898781;
    --grid: #e1e0d9; --border: rgba(11,11,11,0.10);
  }}
  @media (prefers-color-scheme: dark) {{
    :root:not([data-theme="light"]) {{
      color-scheme: dark;
      --surface: #1a1a19; --plane: #0d0d0d;
      --primary: #ffffff; --secondary: #c3c2b7; --muted: #898781;
      --grid: #2c2c2a; --border: rgba(255,255,255,0.10);
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0; padding: 32px 24px 64px;
    background: var(--plane); color: var(--primary);
    font-family: {font};
    font-size: 15px; line-height: 1.5;
  }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 22px; font-weight: 600; margin: 0 0 4px; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 0 0 2px; }}
  .sub {{ color: var(--secondary); font-size: 14px; margin: 0 0 28px; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 20px 12px; margin-bottom: 20px;
  }}
  .card-note {{ color: var(--muted); font-size: 13px; margin: 0 0 12px; }}
  .hero-row {{ display: flex; gap: 20px; flex-wrap: wrap; align-items: stretch; }}
  .hero {{ padding-right: 24px; border-right: 1px solid var(--border); }}
  .hero-value {{ font-size: 48px; font-weight: 600; line-height: 1.1;
                 color: var(--primary); }}
  .tiles {{ display: flex; gap: 32px; flex-wrap: wrap; align-items: center; }}
  .tile-label {{ color: var(--muted); font-size: 13px; }}
  .tile-value {{ font-size: 24px; font-weight: 600; color: var(--primary); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px;
           font-variant-numeric: tabular-nums; }}
  th, td {{ text-align: right; padding: 8px 10px;
            border-bottom: 1px solid var(--grid); }}
  thead th {{ color: var(--muted); font-weight: 500; }}
  tbody th {{ text-align: left; font-weight: 500; color: var(--primary); }}
  tbody tr:last-child th, tbody tr:last-child td {{ border-bottom: none; }}
  .plot {{ overflow-x: auto; }}
  footer {{ color: var(--muted); font-size: 13px; margin-top: 24px; }}
</style>
</head>
<body>
<main>
  <h1>Video duration distribution — TREC 2026 VQA Task 1</h1>
  <p class="sub">{n_videos} videos on disk serving {n_queries} queries ·
     measured with ffprobe · generated {generated}</p>

  <div class="card">{tiles}</div>

  <div class="card">
    <h2>Where the corpus sits</h2>
    <p class="card-note">Counts per 2-second bin, stacked by source.</p>
    <div class="plot">{histogram}</div>
  </div>

  <div class="card">
    <h2>The full range, including the tail</h2>
    <p class="card-note">Cumulative share of videos at or below each duration.
       A log axis so 1 s and 389 s fit the same view.</p>
    <div class="plot">{ecdf}</div>
  </div>

  <div class="card">
    <h2>Summary statistics</h2>
    <p class="card-note">The table view — every plotted value, readable without
       the charts.</p>
    {table}
  </div>

  <footer>Durations read from the container, falling back to the video stream.
    Query-weighted footage counts a video once per query that asks about it.</footer>
</main>
<script>
  const THEMES = {themes};

  // Plotly bakes colours into the figure, so the theme swap has to restyle the
  // traces rather than ride on the CSS variables above.
  function applyTheme(mode) {{
    const t = THEMES[mode];
    if (!window.Plotly) return;

    for (const id of ["hist", "ecdf"]) {{
      const el = document.getElementById(id);
      if (!el || !el.data) continue;

      Plotly.relayout(el, {{
        "paper_bgcolor": t.surface,
        "plot_bgcolor": t.surface,
        "font.color": t.secondary,
        "xaxis.gridcolor": t.grid,
        "yaxis.gridcolor": t.grid,
        "xaxis.linecolor": t.axis,
        "yaxis.linecolor": t.axis,
        "xaxis.tickcolor": t.axis,
        "yaxis.tickcolor": t.axis,
        "xaxis.zerolinecolor": t.axis,
        "yaxis.zerolinecolor": t.axis,
        "xaxis.tickfont.color": t.muted,
        "yaxis.tickfont.color": t.muted,
        "xaxis.title.font.color": t.secondary,
        "yaxis.title.font.color": t.secondary,
        "legend.font.color": t.secondary,
        "hoverlabel.bordercolor": t.axis
      }});

      const n = el.data.length;
      if (id === "hist") {{
        Plotly.restyle(el, {{
          "marker.color": t.series.slice(0, n),
          "marker.line.color": new Array(n).fill(t.surface)
        }});
      }} else {{
        Plotly.restyle(el, {{"line.color": t.series.slice(0, n)}});
      }}

      const notes = (el.layout.annotations || []).map((a, i) => i);
      for (const i of notes) {{
        const colour = el.layout.annotations[i].showarrow ? t.secondary : t.muted;
        Plotly.relayout(el, {{
          ["annotations[" + i + "].font.color"]: colour,
          ["annotations[" + i + "].arrowcolor"]: t.axis
        }});
      }}
    }}
  }}

  const query = window.matchMedia("(prefers-color-scheme: dark)");
  const sync = () => applyTheme(query.matches ? "dark" : "light");
  query.addEventListener("change", sync);
  sync();
</script>
</body>
</html>
"""

HTML_TEMPLATE = HTML_TEMPLATE.replace("{font}", FONT)
