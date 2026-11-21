"""Summarize the step-1 audit: per-replica grokking events, a table per condition, and two figures.

  python analyze_audit.py runs/audit        -> runs/audit/SUMMARY.md, curves.png, collapse.png

Per-replica events (all in optimizer steps, read from the logs taken every `log_every` steps):
  t_fit     first logged step with train accuracy >= 0.99
  t_first   first logged step with test accuracy >= 0.95
  t_stable  first logged step after which test accuracy never drops below 0.95 again
  dips      how many times test accuracy falls below 0.90 after t_first
A replica that never reaches 0.95 is "not grokked by the budget" (right-censored), not "never groks".
Medians treat censored replicas as +infinity (so a median can read ">budget"); they are not medians of
the grokked replicas only, which would make a condition that rarely groks look fast.
"""
import json
import pathlib
import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["ce32", "ce32_ulp", "margin32", "ce64", "ref64"]
COLORS = {"ce32": "#d62728", "ce32_ulp": "#ff9896", "margin32": "#1f77b4", "ce64": "#2ca02c", "ref64": "#7f7f7f"}


def first_index(mask):
    idx = np.flatnonzero(mask)
    return int(idx[0]) if len(idx) else None


def events(step, train_acc, test_acc):
    i_fit = first_index(train_acc >= 0.99)
    i_first = first_index(test_acc >= 0.95)
    below = np.flatnonzero(test_acc < 0.95)
    i_stable = None
    if test_acc[-1] >= 0.95:
        i_stable = int(below[-1]) + 1 if len(below) else 0
    dips, min_after = 0, None
    if i_first is not None:
        after = test_acc[i_first:]
        dips = int(np.sum((after[1:] < 0.90) & (after[:-1] >= 0.90)))
        min_after = float(after.min())
    s = lambda i: None if i is None else int(step[i])
    return dict(t_fit=s(i_fit), t_first=s(i_first), t_stable=s(i_stable), dips=dips, min_after=min_after)


def load(run_dir):
    rows, curves = [], {}
    for v in ORDER:
        npz, js = run_dir / f"{v}.npz", run_dir / f"{v}.json"
        if not npz.exists():
            continue
        L, meta = np.load(npz), json.load(open(js))
        step = L["step"]
        for r, rep in enumerate(meta["replicas"]):
            ev = events(step, L["train_acc"][:, r], L["test_acc"][:, r])
            rows.append(dict(variant=v, wd=rep["wd"], eps=rep["eps"], seed=rep["init_seed"], **ev,
                             final_test=float(L["test_acc"][-1, r]),
                             max_collapse=float(L["collapse_frac"][:, r].max()),
                             max_collapse64=float(L["collapse64_frac"][:, r].max()) if "collapse64_frac" in L.files else None,
                             max_margin=float(L["margin_max"][:, r].max()) if "margin_max" in L.files else None,
                             final_collapse=float(L["collapse_frac"][-1, r]),
                             final_margin_med=float(L["margin_med"][-1, r]),
                             final_l2=float(np.hypot(L["l2_in"][-1, r], L["l2_out"][-1, r])),
                             final_linf=float(max(L["linf_in"][-1, r], L["linf_out"][-1, r]))))
            curves[(v, rep["wd"], rep["eps"], rep["init_seed"])] = {k: L[k][:, r] for k in L.files if k != "step"}
        curves[("step", v)] = step
        rows[-1]["_meta"] = meta
    return rows, curves


def med(xs):
    """Median with censored (None) values counted as +infinity."""
    vals = np.array([np.inf if x is None else x for x in xs], dtype=float)
    m = np.median(vals)
    return ">budget" if not np.isfinite(m) else f"{m:.0f}"


def table(rows):
    lines = ["| variant | wd | eps | grokked (≥0.95 by budget) | median t_first | median t_stable | dips | "
             "min test after first | final test | max fp32-collapse share* | final fp32-collapse share* | "
             "max fp64-collapse share* | final margin (median) | max margin |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    keys = sorted({(r["variant"], r["wd"], r["eps"]) for r in rows},
                  key=lambda k: (k[1], k[2], ORDER.index(k[0])))
    for v, wd, eps in keys:
        g = [r for r in rows if (r["variant"], r["wd"], r["eps"]) == (v, wd, eps)]
        n_grok = sum(r["t_first"] is not None for r in g)
        mins = [r["min_after"] for r in g if r["min_after"] is not None]
        cells = [v, f"{wd:g}", f"{eps:g}", f"{n_grok}/{len(g)}",
                 med([r["t_first"] for r in g]), med([r["t_stable"] for r in g]),
                 str(sum(r["dips"] for r in g)), f"{min(mins):.3f}" if mins else "–",
                 f"{np.mean([r['final_test'] for r in g]):.3f}",
                 f"{np.median([r['max_collapse'] for r in g]):.3f}",
                 f"{np.median([r['final_collapse'] for r in g]):.3f}",
                 f"{np.median([r['final_margin_med'] for r in g]):.1f}",
                 "–" if g[0]["max_collapse64"] is None else f"{np.median([r['max_collapse64'] for r in g]):.3f}",
                 "–" if g[0]["max_margin"] is None else f"{max(r['max_margin'] for r in g):.1f}"]
        cells[-2], cells[-3] = cells[-3], cells[-2]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def figure(rows, curves, key, ylabel, path, ylim=None, hline=None):
    wds = sorted({r["wd"] for r in rows})
    epss = sorted({r["eps"] for r in rows})
    fig, axes = plt.subplots(len(wds), len(epss), figsize=(4.2 * len(epss), 3.0 * len(wds)),
                             sharex=True, sharey=True, squeeze=False)
    for i, wd in enumerate(wds):
        for j, eps in enumerate(epss):
            ax = axes[i, j]
            for v in ORDER:
                for r in rows:
                    if (r["variant"], r["wd"], r["eps"]) != (v, wd, eps):
                        continue
                    c = curves[(v, wd, eps, r["seed"])]
                    ax.plot(curves[("step", v)], c[key], color=COLORS[v], lw=0.9, alpha=0.7,
                            label=v if r["seed"] == 0 else None)
            if hline is not None:
                ax.axhline(hline, color="k", lw=0.6, ls=":")
            ax.set_title(f"wd={wd:g}, eps={eps:g}", fontsize=9)
            if ylim:
                ax.set_ylim(*ylim)
            if i == len(wds) - 1:
                ax.set_xlabel("step")
            if j == 0:
                ax.set_ylabel(ylabel)
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h); labels.append(l)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(run_dir):
    run_dir = pathlib.Path(run_dir)
    rows, curves = load(run_dir)
    metas = {r["variant"]: r.pop("_meta") for r in rows if "_meta" in r}
    for r in rows:
        r.pop("_meta", None)
    figure(rows, curves, "test_acc", "test accuracy", run_dir / "curves.png", ylim=(0, 1.02), hline=0.95)
    figure(rows, curves, "collapse_frac", "train share in fp32 collapse", run_dir / "collapse.png", ylim=(0, 1.02))
    figure(rows, curves, "margin_med", "median train margin", run_dir / "margin.png", hline=15.4)
    devices = {(m["device"], m["device_name"]) for m in metas.values()}
    if len(devices) > 1:
        print("WARNING: variants ran on different devices:", devices)
    env = "\n".join(f"- {v}: {m['device']} {m['device_name']}, torch {m['torch']}, {m['seconds']} s, "
                    f"complete {m.get('complete')}, non-finite loss at logged step {m.get('nonfinite_loss_at_log_step')}" for v, m in metas.items())
    md = f"# Step-1 numerics audit\n\n{env}\n\n{table(rows)}\n\nFigures: curves.png (test accuracy), " \
         f"collapse.png (share of train examples whose float32 softmax_y == 1), margin.png (median train margin; " \
         f"dotted line = margin 15.4, where float32 collapse was seen to start).\n\n" \
         f"*Collapse shares are computed from each run's logits. Only for ce32/ce32_ulp do they describe " \
         f"the gradient the run actually used; for the other variants they say what standard cross-entropy " \
         f"would have done with those logits. Medians count runs not grokked by the budget as +infinity.\n"
    (run_dir / "SUMMARY.md").write_text(md)
    json.dump(rows, open(run_dir / "replicas.json", "w"), indent=1)
    print(md)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/audit")
