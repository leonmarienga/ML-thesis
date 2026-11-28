"""Build the F9 view and T9 extract from immutable Batch 14 summaries only."""
import csv
import hashlib
import math
from pathlib import Path
import sys

DIGESTS = {
    "pareto_points.csv": "db1e918336973b1bf181561e9847f7eebd474739ec70a5d349a126c9b044e73a",
    "preference_choices.csv": "603092e3e624d34fa0543ae7ee076ebe94f9e091f187837af1a92c2921508eee",
    "disruption_sensitivity.csv": "543486f72b285004582320837dbb058b018d078e5da78bc2c26849e2b4c06b91",
}
KEYS = ("overall", "high_vulnerability", "least_served")
OPTIONS = {
    "total service": ("A", (1, 0, 0), 34, -12),
    "vulnerability priority": ("B", (.25, .65, .10), 21, 23),
    "minimum coverage priority": ("C", (.20, .20, .60), -30, -26),
    "balanced": ("D", (.40, .30, .30), 10, -32),
}
REGIMES = (("normal", 15), ("delay", 18), ("closure", 15))


def build(source, destination):
    def read(name):
        path = source / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != DIGESTS[name]:
            raise ValueError(f"Input digest mismatch: {name}")
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    rows = read("pareto_points.csv")
    prefs = read("preference_choices.csv")
    conditions = read("disruption_sensitivity.csv")
    ref = {(row["policy"], row["floor"]): row for row in rows}
    metrics = lambda r: tuple(float(r[k]) for k in KEYS)
    eps = 1e-9
    clear = lambda r: not any(
        all(x >= y - eps for x, y in zip(metrics(t), metrics(r)))
        and any(x > y + eps for x, y in zip(metrics(t), metrics(r)))
        for t in rows
    )
    if len(rows) != 89 or len(ref) != 89 or sum(clear(r) for r in rows) != 46:
        raise ValueError("Mismatching frontier size")
    if not all(clear(r) == (r["nondominated"] == "True") for r in rows):
        raise ValueError("Mismatching frontier identity")
    if [r["preference"] for r in prefs] != list(OPTIONS):
        raise ValueError("Mismatching preference registry")
    for p in prefs:
        _, weights, _, _ = OPTIONS[p["preference"]]
        score = lambda r: sum(w * v for w, v in zip(weights, metrics(r)))
        item = ref[(p["policy"], p["floor"])]
        if (not clear(item) or metrics(p) != metrics(item)
                or not math.isclose(score(item), float(p["score"]), abs_tol=1e-12)
                or not math.isclose(score(item), max(score(r) for r in rows if clear(r)), abs_tol=1e-12)):
            raise ValueError("Preference reproduction mismatch")
    condition_keys = {(r["regime"], r["policy"], r["floor"]): r for r in conditions}
    floor_results = [(regime, n, *[metrics(condition_keys[regime, "vulnerability", f])[2] * 100
                                   for f in ("0.0", "1.0")]) for regime, n in REGIMES]

    destination.mkdir(parents=True, exist_ok=True)
    (destination / "transport").mkdir(exist_ok=True)
    with (destination / "transport" / "T9-preference-summary.csv").open("w", newline="", encoding="utf-8") as port:
        writer = csv.writer(port, lineterminator="\n")
        writer.writerow(("marker", "preference", "policy", "floor_pct", "total_pct", "high_vulnerability_pct", "least_served_pct"))
        for p in prefs:
            writer.writerow((OPTIONS[p["preference"]][0], p["preference"], p["policy"], f'{float(p["floor"])*100:.0f}',
                             *(f"{v*100:.2f}" for v in metrics(p))))

    chart = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 430" role="img" aria-labelledby="title desc">',
             '<title id="title">Simulated transport policy trade-offs and geographic floor by road regime</title>',
             '<desc id="desc">Left: all 89 settings projected onto least-served and high-vulnerability coverage. Frontier status uses all three objectives, including total service. Right: the minimum-zone gains from a full floor vary by transport regime.</desc>',
             '<style>text{font-family:DejaVu Sans,sans-serif;fill:#183049;font-size:15px}.h{font-size:19px;font-weight:bold}.t{font-size:13px}.v{font-weight:bold}.grid{stroke:#d8e0ea;stroke-width:1}.axis{stroke:#496581;stroke-width:1.4}</style>',
             '<rect width="960" height="430" fill="#fff"/>']
    def label(x, y, content, cls="", anchor="start"):
        chart.append(f'<text x="{x:g}" y="{y:g}" text-anchor="{anchor}" class="{cls}">{content}</text>')
    def line(x, y, a, b, cls="grid"):
        chart.append(f'<path d="M{x:g} {y:g} L{a:g} {b:g}" class="{cls}"/>')
    def dot(x, y, r, color, stroke="#fff"):
        chart.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:g}" fill="{color}" stroke="{stroke}"/>')

    label(22, 31, "A. Policy trade-offs", "h")
    label(530, 31, "B. Geographic floor under disruption", "h")
    label(22, 55, "High-vulnerability service (%)", "t")
    label(530, 55, "Mean least-served-zone coverage (%)", "t")
    xpos = lambda x: 62 + x * 8.75
    ypos = lambda y: 322 - (y - 30) * 4.72
    for x in range(0, 46, 10):
        line(xpos(x), 86, xpos(x), 322)
        label(xpos(x), 342, x, "t", "middle")
    for y in range(30, 81, 10):
        line(62, ypos(y), 456, ypos(y))
        label(53, ypos(y) + 5, y, "t", "end")
    line(62, 86, 62, 322, "axis")
    line(62, 322, 456, 322, "axis")
    for r in sorted(rows, key=lambda p: p["nondominated"]):
        _, v, m = metrics(r)
        dot(xpos(m * 100), ypos(v * 100), 4.1, "#147d92" if r["nondominated"] == "True" else "#aebdcb")
    for row in prefs:
        mark, _, dx, dy = OPTIONS[row["preference"]]
        _, v, m = metrics(row)
        x, y = xpos(m * 100), ypos(v * 100)
        line(x, y, x+dx, y+dy-5, "axis")
        dot(x, y, 5, "#8f3a21")
        label(x+dx, y+dy, mark, "v", "middle")
    label(253, 365, "Least-served-zone coverage (%)", "", "middle")
    dot(72, 392, 5, "#147d92"); label(84, 397, "Nondominated (3 objectives)", "t")
    dot(306, 392, 5, "#aebdcb"); label(318, 397, "Dominated", "t")
    rpos = lambda x: 624 + x * 4.8
    for x in range(0, 61, 10):
        line(rpos(x), 86, rpos(x), 322)
        label(rpos(x), 342, x, "t", "middle")
    for i, (regime, n, low, high) in enumerate(floor_results):
        y = 124 + 72 * i
        label(608, y + 4, f"{regime.title()} ({n})", "t", "end")
        line(rpos(low), y, rpos(high), y, "axis")
        dot(rpos(low), y, 6, "#fff", "#8f3a21")
        dot(rpos(high), y, 6, "#8f3a21")
        label(rpos(low), y-13, f"{low:.1f}", "t", "middle")
        label(rpos(high), y-13, f"{high:.1f}", "t", "middle")
        label((rpos(low)+rpos(high))/2, y+22, f"+{high-low:.1f} pp", "t", "middle")
    line(624, 322, 912, 322, "axis")
    dot(662, 392, 6, "#fff", "#8f3a21"); label(674, 397, "No floor", "t")
    dot(779, 392, 6, "#8f3a21"); label(791, 397, "Full feasible floor", "t")
    chart.append("</svg>")
    (destination / "F9-simulated-transport.svg").write_text("\n".join(chart)+"\n", encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: build_transport_summary.py <pinned CSV dir> <output figures dir>")
    build(Path(sys.argv[1]), Path(sys.argv[2]))
