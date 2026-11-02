#!/usr/bin/env python3
"""Paper 7 V110: minimax design for delays 0/6/12.
Measured expression knots only; no biological delay truth.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, sys, traceback, zipfile, shutil, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_three_delay_design_v110" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
GENES = (
    "PRL", "IGFBP1", "FOXO1", "HAND2",
    "WNT4", "PGR", "IL11", "LEFTY2"
)
CULTURES = ("S169", "S170", "S506", "S508")
BASE = (0, 12, 24)
SCHEDULES = {
    "base_0_12_24": BASE,
    **{
        "plus_" + str(t): tuple(sorted(BASE + (t,)))
        for t in (3, 6, 9, 18)
    }
}
PAIRS = ((0, 6), (0, 12), (6, 12))
SOURCE_SHA = (
    "eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a"
)

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def digest(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def save(name, rows):
    d = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    d.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | rows={}".format(name, len(d)))
    return d

def load():
    base = ROOT / "paper7_shift_specific_design_v109"
    required = (
        "11_sha256_inventory.csv",
        "10_manifest.json",
        "02_shift_specific_template_distances.csv",
        "04_own_vs_unknown_culture.csv",
        "05_shift_specific_winners.csv",
        "input_source_v67_measured.csv"
    )
    runs = sorted(
        (
            r for r in base.glob("run_*")
            if r.is_dir() and all((r / f).is_file() for f in required)
        ),
        reverse=True
    )
    need(runs, "Complete V109 archive unavailable")
    run = runs[0]

    inv = pd.read_csv(
        run / "11_sha256_inventory.csv",
        encoding="utf-8-sig"
    )
    need(
        {"filename", "bytes", "sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "V109 inventory malformed"
    )
    for r in inv.itertuples(index=False):
        name = str(r.filename)
        p = run / name
        need(
            Path(name).name == name
            and p.is_file()
            and p.stat().st_size == int(r.bytes)
            and digest(p) == str(r.sha256),
            "Archived V109 file changed: " + name
        )

    manifest = json.loads(
        (run / "10_manifest.json").read_text("utf-8")
    )
    need(
        manifest.get("version") == "v109"
        and manifest.get("comparison_rows") == 160,
        "V109 manifest mismatch"
    )

    source = run / "input_source_v67_measured.csv"
    need(digest(source) == SOURCE_SHA, "Measured source SHA256 differs")

    raw = pd.read_csv(source, encoding="utf-8-sig")
    need(
        len(raw) == 320
        and not raw.duplicated(
            ["culture_label", "gene", "time_hours"]
        ).any(),
        "Unexpected source samples"
    )

    curves = {}
    for culture in CULTURES:
        for gene in GENES:
            d = raw[
                (raw.culture_label == culture)
                & (raw.gene == gene)
            ].sort_values("time_hours")
            tt = d.time_hours.to_numpy(float)
            yy = d.log2CPM.to_numpy(float)
            need(
                len(tt) == 10
                and np.allclose(
                    tt, [0,3,6,9,12,18,24,36,48,96]
                )
                and np.isfinite(yy).all(),
                "Measured time grid mismatch"
            )
            curves[(culture, gene)] = {
                int(t): float(y) for t, y in zip(tt, yy)
            }

    old = pd.read_csv(
        run / "02_shift_specific_template_distances.csv",
        encoding="utf-8-sig"
    )
    prior = pd.read_csv(
        run / "04_own_vs_unknown_culture.csv",
        encoding="utf-8-sig"
    )
    need(
        len(old) == 160 and len(prior) == 40,
        "V109 archived comparisons incomplete"
    )

    say(
        "[V109 VERIFIED] {} | {} files | 32 measured curves"
        .format(run, len(inv))
    )
    return run, curves, old, prior

def vec(curve, grid, tau):
    hours = [0 if t <= tau else t-tau for t in grid]
    need(
        all(t in curve for t in hours),
        "Hypothetical shift requires unmeasured knot"
    )
    return np.asarray([curve[t] for t in hours], float)

def sse(y, x):
    yc = y-y.mean()
    xc = x-x.mean()
    denominator = float(xc @ xc)
    amplitude = (
        max(0., float(yc @ xc) / denominator)
        if denominator > 1e-14 else 0.
    )
    residual = yc-amplitude*xc
    return float(residual @ residual)

def calculate(curves, old):
    rows, genes = [], []
    lookup = old.set_index([
        "delay", "schedule",
        "culture_target", "culture_alternative"
    ])
    replay = 0

    for schedule, grid in SCHEDULES.items():
        for ca in CULTURES:
            for cb in CULTURES:
                for ta, tb in PAIRS:
                    forward = reverse = 0.

                    for gene in GENES:
                        a = curves[(ca, gene)]
                        b = curves[(cb, gene)]

                        f = sse(
                            vec(a, grid, ta),
                            vec(b, grid, tb)
                        )
                        r = sse(
                            vec(a, grid, tb),
                            vec(b, grid, ta)
                        )
                        forward += f
                        reverse += r

                        genes.append(dict(
                            schedule=schedule,
                            culture_target=ca,
                            culture_alternative=cb,
                            tau_a=ta,
                            tau_b=tb,
                            gene=gene,
                            forward_SSE=f,
                            reverse_SSE=r
                        ))

                    minimum = min(forward, reverse)

                    rows.append(dict(
                        schedule=schedule,
                        extra_hours=len(grid)-3,
                        culture_target=ca,
                        culture_alternative=cb,
                        tau_a=ta,
                        tau_b=tb,
                        pair="{}v{}".format(ta, tb),
                        forward_SSE=forward,
                        reverse_SSE=reverse,
                        min_SSE=minimum,
                        matched_culture=ca == cb
                    ))

                    if ta == 0:
                        original = lookup.loc[
                            (tb, schedule, ca, cb)
                        ]
                        need(
                            abs(float(original.min_SSE)-minimum) < 1e-8,
                            "V109 pairwise replay failed"
                        )
                        replay += 1

        say("[PAIRWISE] {} | all three delay pairs".format(schedule))

    need(replay == 160, "Incomplete historical replay")
    say("[V109 EXACT REPLAY] 160/160 pairwise SSE")

    return (
        save("02_three_pair_distances.csv", rows),
        save("03_gene_direction_SSE.csv", genes)
    )

def compare(dist, prior):
    replay = 0
    for (ca, schedule, ta, tb), group in (
        dist[dist.tau_a == 0].groupby([
            "culture_target", "schedule", "tau_a", "tau_b"
        ])
    ):
        own = float(
            group[group.matched_culture].iloc[0].min_SSE
        )
        other = float(
            group[~group.matched_culture].min_SSE.min()
        )
        old = prior[
            (prior.culture == ca)
            & (prior.schedule == schedule)
            & (prior.delay == tb)
        ]
        need(
            len(old) == 1
            and abs(own-float(old.iloc[0].own_SSE)) < 1e-8
            and abs(other-float(old.iloc[0].nearest_other_SSE)) < 1e-8,
            "V109 culture-level replay mismatch"
        )
        replay += 1

    need(replay == 40, "Historical culture replay incomplete")
    say("[V109 CULTURE REPLAY] 40/40 exact")

    bottlenecks, pairwise = [], []

    for culture in CULTURES:
        for schedule in SCHEDULES:
            sub = dist[
                (dist.culture_target == culture)
                & (dist.schedule == schedule)
            ]
            need(len(sub) == 12, "Incomplete three-pair library")

            for objective, part in (
                ("known_culture", sub[sub.matched_culture]),
                ("unknown_culture", sub)
            ):
                first = part.sort_values([
                    "min_SSE", "pair", "culture_alternative"
                ]).iloc[0]

                bottlenecks.append(dict(
                    culture=culture,
                    schedule=schedule,
                    objective=objective,
                    min_three_pair_SSE=float(first.min_SSE),
                    bottleneck_pair=first.pair,
                    bottleneck_culture=first.culture_alternative
                ))

            for pair, g in sub.groupby("pair"):
                pairwise.append(dict(
                    culture=culture,
                    schedule=schedule,
                    pair=pair,
                    known_culture_min_SSE=float(
                        g[g.matched_culture].min_SSE.iloc[0]
                    ),
                    unknown_culture_min_SSE=float(
                        g.min_SSE.min()
                    )
                ))

    b = save("04_three_pair_bottlenecks.csv", bottlenecks)
    save("05_pairwise_design_tradeoffs.csv", pairwise)

    winners = []
    for (culture, objective), g in b.groupby(
        ["culture", "objective"]
    ):
        base = float(
            g[g.schedule == "base_0_12_24"]
            .min_three_pair_SSE.iloc[0]
        )
        options = g[
            g.schedule != "base_0_12_24"
        ].sort_values(
            ["min_three_pair_SSE", "schedule"],
            ascending=[False, True]
        )
        best = options.iloc[0]
        runner = options.iloc[1]

        need(
            best.min_three_pair_SSE + 1e-8 >= base,
            "Nested-design minimax SSE decreased"
        )

        winners.append(dict(
            culture=culture,
            objective=objective,
            selected=best.schedule,
            bottleneck_pair=best.bottleneck_pair,
            bottleneck_culture=best.bottleneck_culture,
            gain_over_base=float(
                best.min_three_pair_SSE-base
            ),
            margin_over_runner=float(
                best.min_three_pair_SSE
                -runner.min_three_pair_SSE
            )
        ))

    w = save("06_three_pair_best_one_extra.csv", winners)

    global_rows = []
    for (objective, schedule), g in b.groupby([
        "objective", "schedule"
    ]):
        limiting = g.loc[g.min_three_pair_SSE.idxmin()]
        global_rows.append(dict(
            objective=objective,
            schedule=schedule,
            global_worst_SSE=float(g.min_three_pair_SSE.min()),
            mean_culture_worst_SSE=float(
                g.min_three_pair_SSE.mean()
            ),
            limiting_culture=limiting.culture
        ))

    global_table = save(
        "07_global_minimax_design.csv", global_rows
    )

    for r in w.itertuples(index=False):
        say(
            "[V110 WINNER | {} | {}] {} | bottleneck={} / {} "
            "| gain={:.4f}".format(
                r.culture, r.objective, r.selected,
                r.bottleneck_pair, r.bottleneck_culture,
                r.gain_over_base
            )
        )

    for objective, g in global_table.groupby("objective"):
        best = g[
            g.schedule != "base_0_12_24"
        ].sort_values(
            ["global_worst_SSE", "schedule"],
            ascending=[False, True]
        ).iloc[0]
        say(
            "[V110 GLOBAL | {}] {} | minimum SSE={:.5f}"
            .format(
                objective, best.schedule, best.global_worst_SSE
            )
        )
    return b

def gene_loo(components):
    lookup = {}
    for keys, g in components.groupby([
        "schedule", "culture_target",
        "culture_alternative", "tau_a", "tau_b"
    ]):
        need(
            len(g) == 8 and set(g.gene) == set(GENES),
            "Gene decomposition incomplete"
        )
        lookup[keys] = g.set_index("gene")

    rows = []
    for culture in CULTURES:
        for deleted in GENES:
            for schedule in SCHEDULES:
                minimum = np.inf

                for alternative in CULTURES:
                    for ta, tb in PAIRS:
                        g = lookup[
                            (schedule,culture,alternative,ta,tb)
                        ].drop(index=deleted)

                        distance = min(
                            float(g.forward_SSE.sum()),
                            float(g.reverse_SSE.sum())
                        )
                        minimum = min(minimum, distance)

                rows.append(dict(
                    culture=culture,
                    deleted_gene=deleted,
                    schedule=schedule,
                    unknown_culture_min_SSE=minimum
                ))

    result = save(
        "08_leave_one_gene_out_all_options.csv", rows
    )

    choices = []
    for (culture, deleted), g in result.groupby([
        "culture", "deleted_gene"
    ]):
        options = g[
            g.schedule != "base_0_12_24"
        ].sort_values(
            ["unknown_culture_min_SSE", "schedule"],
            ascending=[False, True]
        )
        choices.append(dict(
            culture=culture,
            deleted_gene=deleted,
            selected=options.iloc[0].schedule
        ))

    selected = save(
        "09_leave_one_gene_out_winners.csv", choices
    )

    for hour in (3,6,9,18):
        name = "plus_"+str(hour)
        say(
            "[V110 LOO] {} selected in {}/32 removals"
            .format(name, int((selected.selected == name).sum()))
        )

def loco(bottlenecks):
    subset = bottlenecks[
        bottlenecks.objective == "unknown_culture"
    ]
    matrix = subset.pivot(
        index="culture",
        columns="schedule",
        values="min_three_pair_SSE"
    )
    rows = []
    for culture in CULTURES:
        training = matrix.drop(index=culture)[
            ["plus_3","plus_6","plus_9","plus_18"]
        ].mean(axis=0)
        test = matrix.loc[culture, [
            "plus_3","plus_6","plus_9","plus_18"
        ]]
        choice = training.idxmax()
        best = test.idxmax()

        rows.append(dict(
            held_out_culture=culture,
            selected_from_three=choice,
            heldout_best=best,
            agreement=choice == best,
            regret_SSE=float(test.max()-test[choice])
        ))

    df = save(
        "10_leave_one_culture_out_descriptive.csv", rows
    )
    say(
        "[V110 LOCO] selected time agrees in {}/4 cultures "
        "(descriptive only)".format(int(df.agreement.sum()))
    )

def main():
    say("="*89)
    say("PAPER 7 | V110 | MINIMAX DESIGN FOR 0/6/12h")
    say("MEASURED SOURCE | NO INTERPOLATION | NO DELAY TRUTH | NO GIT")
    say("="*89)

    need(len(sys.argv) == 1, "Run without flags")
    run, curves, old, prior = load()

    protocol = dict(
        version="v110",
        delay_pairs=PAIRS,
        schedules=SCHEDULES,
        source_sha256=SOURCE_SHA,
        criterion=(
            "Maximize minimum profiled SSE over three delay "
            "pairs and four alternative cultures"
        ),
        nuisance="Gene-specific intercept and nonnegative amplitude",
        delayed_samples_observed=False,
        noise_sigma_available=False,
        no_interpolation=True,
        no_git=True,
        script_sha256=digest(__file__)
    )
    (OUT/"01_protocol.json").write_text(
        json.dumps(protocol,indent=2)+"\n",
        encoding="utf-8"
    )

    distances, components = calculate(curves, old)
    bottlenecks = compare(distances, prior)
    gene_loo(components)
    loco(bottlenecks)

    (OUT/"11_methods_limitations.md").write_text(
        """# Paper 7 V110 — three-delay minimax design

Compares 0/6, 0/12 and 6/12-hour hypothetical shifts.

Only directly measured source timepoints are reindexed.
No delayed biological samples are available.

The alternative response has a flat pre-onset period.
Each gene has a free intercept and nonnegative amplitude.

Both shift directions are evaluated. Gene-wise SSE is
summed before choosing the smaller direction.

The unknown-culture objective minimizes over all
four source-culture alternatives and all three pairs.
It is not a validated classification-risk estimate.

The choice is conditional on shifted-copy assumptions.
Arbitrary smooth response shapes can still produce
structural onset ambiguities.

At 6h, tau6 and tau12 both predict pre-response
baseline; 6h may still help constrain nuisance
parameters but does not directly separate the two.

Leave-one-gene and leave-one-culture analyses
are internal sensitivity checks, not external validation.

No replicate-derived error SD or covariance is available.
No claim of practical detection power is made.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT/"script_snapshot_v110.py"
    )
    for src, dst in (
        ("10_manifest.json","input_v109_manifest.json"),
        ("02_shift_specific_template_distances.csv",
         "input_v109_pairwise.csv"),
        ("input_source_v67_measured.csv",
         "input_measured_source.csv")
    ):
        shutil.copy2(run/src,OUT/dst)

    (OUT/"12_manifest.json").write_text(
        json.dumps(dict(
            version="v110",
            script_sha256=digest(__file__),
            source_sha256=SOURCE_SHA,
            pairwise_rows=len(distances),
            gene_direction_rows=len(components),
            replayed_V109_pairs=160,
            no_delay_truth=True,
            noise_unavailable=True,
            python=sys.version,
            numpy=np.__version__,
            pandas=pd.__version__,
            platform=platform.platform(),
            no_git=True
        ),indent=2)+"\n",
        encoding="utf-8"
    )

    say(
        "[BOUNDARY] Model-conditional geometry only; "
        "not experimental detection validity"
    )
    say("[COMPLETE] Three-pair design and robustness archived")

def finish(ok):
    say("[RESULT] "+("SUCCESS" if ok else "FAILED"))
    say("[REPORT] "+str(OUT/"console_summary.txt"))
    archive = OUT/"portable_report_v110.zip"
    say("[ZIP] "+str(archive))
    (OUT/"console_summary.txt").write_text(
        "\n".join(LOG)+"\n",encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            archive.name,"13_sha256_inventory.csv"
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT/"13_sha256_inventory.csv",
        index=False,encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

if __name__ == "__main__":
    ok = True
    try:
        main()
    except Exception:
        ok = False
        say("[ERROR]\n"+traceback.format_exc())
    finally:
        finish(ok)
    if not ok:
        sys.exit(1)
