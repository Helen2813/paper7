#!/usr/bin/env python3
"""Paper 7 V109: shift-specific design and unknown-culture alternatives.
Measured source knots only. No natural delayed biological samples.
"""
from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, traceback, shutil, sys, zipfile, platform
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_shift_specific_design_v109" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
GENES = (
    "PRL","IGFBP1","FOXO1","HAND2",
    "WNT4","PGR","IL11","LEFTY2"
)
CULTURES = ("S169","S170","S506","S508")
BASE = (0,12,24)
EXTRAS = (3,6,9,18)
DELAYS = (6,12)
SOURCE_SHA = "eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a"

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
        for x in iter(lambda:f.read(1048576),b""):
            h.update(x)
    return h.hexdigest()

def save(name, rows):
    x = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    x.to_csv(
        OUT/name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | rows={}".format(name,len(x)))
    return x

def load():
    root = ROOT/"paper7_early_gene_design_v107"
    files = (
        "09_sha256_inventory.csv",
        "08_manifest.json",
        "source_measured_v67.csv",
        "04_equal_cost_conditional_designs.csv"
    )

    runs = sorted(
        (r for r in root.glob("run_*")
         if r.is_dir() and all((r/f).is_file() for f in files)),
        reverse=True
    )
    need(runs, "Complete V107 archive not found")
    run = runs[0]

    inv = pd.read_csv(
        run/"09_sha256_inventory.csv", encoding="utf-8-sig"
    )
    need(
        {"filename","bytes","sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "Bad V107 inventory"
    )

    for rec in inv.itertuples(index=False):
        name = str(rec.filename)
        p = run/name
        need(
            name == Path(name).name
            and p.is_file()
            and p.stat().st_size == int(rec.bytes)
            and digest(p) == str(rec.sha256),
            "V107 modified: "+name
        )

    man = json.loads(
        (run/"08_manifest.json").read_text("utf-8")
    )
    path = run/"source_measured_v67.csv"

    need(
        man.get("version") == "v107"
        and man.get("source_sha256") == SOURCE_SHA
        and digest(path) == SOURCE_SHA,
        "Measured source provenance differs"
    )

    raw = pd.read_csv(path,encoding="utf-8-sig")
    need(
        len(raw) == 320
        and {"culture_label","gene","time_hours","log2CPM"}.issubset(raw.columns),
        "Source dimensions differ"
    )
    need(
        not raw.duplicated([
            "culture_label","gene","time_hours"
        ]).any(),
        "Repeated source knots"
    )

    curves = {}

    for culture in CULTURES:
        for gene in GENES:
            s = raw[
                (raw.culture_label == culture)
                & (raw.gene == gene)
            ].sort_values("time_hours")

            t = s.time_hours.to_numpy(float)
            y = s.log2CPM.to_numpy(float)

            need(
                len(t) == 10
                and np.allclose(
                    t,[0,3,6,9,12,18,24,36,48,96]
                )
                and np.isfinite(y).all(),
                "Invalid trajectory "+culture+"/"+gene
            )

            curves[(culture,gene)] = {
                int(ti):float(yi)
                for ti,yi in zip(t,y)
            }

    prior = pd.read_csv(
        run/"04_equal_cost_conditional_designs.csv",
        encoding="utf-8-sig"
    )
    need(
        len(prior) == 176
        and {
            "schedule","culture_target",
            "culture_alternative","min_SSE"
        }.issubset(prior.columns),
        "V107 comparisons not found"
    )

    say(
        "[V107 VERIFIED] {} | inventory={} | 32 measured trajectories"
        .format(run,len(inv))
    )
    return run,curves,prior

def vec(curve,grid,tau):
    knots = [
        0 if t<=tau else t-tau
        for t in grid
    ]
    need(
        all(k in curve for k in knots),
        "Shift requires an unmeasured time"
    )
    return np.asarray(
        [curve[k] for k in knots],float
    )

def sse(y,x):
    xc = x-x.mean()
    yc = y-y.mean()
    den = float(xc@xc)
    amp = (
        max(0.,float(xc@yc)/den)
        if den>1e-14 else 0.
    )
    err = yc-amp*xc
    return float(err@err)

def evaluate(curves,prior):
    plans = [
        ("base_0_12_24",BASE)
    ] + [
        ("plus_"+str(t),tuple(sorted(BASE+(t,))))
        for t in EXTRAS
    ]

    direct = []
    gene = []
    checked = 0

    for tau in DELAYS:
        for label,grid in plans:
            for ca in CULTURES:
                for cb in CULTURES:
                    f = 0.
                    r = 0.

                    for g in GENES:
                        a = curves[(ca,g)]
                        b = curves[(cb,g)]

                        ff = sse(
                            vec(a,grid,0),
                            vec(b,grid,tau)
                        )
                        rr = sse(
                            vec(a,grid,tau),
                            vec(b,grid,0)
                        )

                        f += ff
                        r += rr

                        if ca == cb:
                            gene.append(dict(
                                delay=tau,
                                schedule=label,
                                culture=ca,
                                gene=g,
                                forward_SSE=ff,
                                reverse_SSE=rr
                            ))

                    row = dict(
                        delay=tau,
                        schedule=label,
                        hours=";".join(map(str,grid)),
                        extra_hours=len(grid)-3,
                        culture_target=ca,
                        culture_alternative=cb,
                        same_culture=ca==cb,
                        forward_SSE=f,
                        reverse_SSE=r,
                        min_SSE=min(f,r),
                        sqrtSSE=np.sqrt(min(f,r)),
                        n_gene_times=8*len(grid)
                    )
                    direct.append(row)

                    if tau == 6:
                        old = prior[
                            (prior.schedule==label)
                            & (prior.culture_target==ca)
                            & (prior.culture_alternative==cb)
                        ]

                        need(
                            len(old)==1
                            and abs(
                                float(old.iloc[0].min_SSE)
                                -row["min_SSE"]
                            )<1e-8,
                            "V107 six-hour replay mismatch: "+
                            label+"/"+ca+"/"+cb
                        )
                        checked += 1

        say(
            "[SHIFT] tau={}h | five equal-cost schedules"
            .format(tau)
        )

    need(checked==80,"Unexpected V107 replay count")
    say("[V107 EXACT REPLAY] 80/80 archived 6h comparisons")

    d = save(
        "02_shift_specific_template_distances.csv",direct
    )
    save(
        "03_same_culture_gene_contributions.csv",gene
    )

    rows = []

    for (tau,culture,grid),g in d.groupby([
        "delay","culture_target","schedule"
    ]):
        same = g[g.same_culture]
        cross = g[
            ~g.same_culture
        ].sort_values([
            "min_SSE","culture_alternative"
        ])

        need(
            len(same)==1 and len(cross)==3,
            "Incomplete cross-culture templates"
        )

        a = float(same.iloc[0].min_SSE)
        b = float(cross.iloc[0].min_SSE)

        rows.append(dict(
            delay=tau,
            culture=culture,
            schedule=grid,
            own_SSE=a,
            nearest_other_SSE=b,
            nearest_other_culture=str(
                cross.iloc[0].culture_alternative
            ),
            robust_min_SSE=min(a,b),
            other_closer=bool(b<a-1e-10)
        ))

    r = save(
        "04_own_vs_unknown_culture.csv",rows
    )

    winners = []
    gains = []

    for (tau,culture),g in r.groupby([
        "delay","culture"
    ]):
        baseline = g[
            g.schedule=="base_0_12_24"
        ].iloc[0]

        options = g[
            g.schedule!="base_0_12_24"
        ]

        need(
            len(options)==4,
            "Incomplete equal-cost options"
        )

        for score in (
            "own_SSE",
            "nearest_other_SSE",
            "robust_min_SSE"
        ):
            best = options.sort_values(
                [score,"schedule"],
                ascending=[False,True]
            ).iloc[0]

            vals = options.set_index(
                "schedule"
            )[score].to_dict()

            winners.append(dict(
                delay=tau,
                culture=culture,
                objective=score,
                selected=best.schedule,
                increment_vs_base=float(
                    best[score]-baseline[score]
                ),
                gain_add_3=float(
                    vals["plus_3"]-baseline[score]
                ),
                gain_add_6=float(
                    vals["plus_6"]-baseline[score]
                ),
                gain_add_9=float(
                    vals["plus_9"]-baseline[score]
                ),
                gain_add_18=float(
                    vals["plus_18"]-baseline[score]
                )
            ))

        for option in options.itertuples(index=False):
            need(
                option.own_SSE+1e-8>=baseline.own_SSE
                and option.nearest_other_SSE+1e-8>=baseline.nearest_other_SSE
                and option.robust_min_SSE+1e-8>=baseline.robust_min_SSE,
                "Nested schedule SSE unexpectedly decreased"
            )

            gains.append(dict(
                delay=tau,
                culture=culture,
                schedule=option.schedule,
                own_gain=(
                    option.own_SSE-baseline.own_SSE
                ),
                nearest_other_gain=(
                    option.nearest_other_SSE
                    -baseline.nearest_other_SSE
                ),
                robust_gain=(
                    option.robust_min_SSE
                    -baseline.robust_min_SSE
                ),
                other_closer=option.other_closer
            ))

    w = save(
        "05_shift_specific_winners.csv",winners
    )
    save(
        "06_equal_cost_gain_by_culture.csv",gains
    )

    summary = w.groupby([
        "delay","objective","selected"
    ]).size().rename("n_cultures").reset_index()

    save(
        "07_selected_time_counts.csv",summary
    )

    flip = []

    for culture in CULTURES:
        for objective in (
            "own_SSE",
            "nearest_other_SSE",
            "robust_min_SSE"
        ):
            p = w[
                (w.culture==culture)
                & (w.objective==objective)
            ].set_index("delay")

            flip.append(dict(
                culture=culture,
                objective=objective,
                choice_6=str(p.loc[6,"selected"]),
                choice_12=str(p.loc[12,"selected"]),
                same=bool(
                    p.loc[6,"selected"]
                    ==p.loc[12,"selected"]
                )
            ))

    flip = save(
        "08_shift_target_specificity.csv",flip
    )

    for tau in DELAYS:
        for objective in (
            "own_SSE",
            "robust_min_SSE"
        ):
            part = w[
                (w.delay==tau)
                & (w.objective==objective)
            ]

            choices = ", ".join(
                str(x.culture)+":"+str(x.selected)
                for x in part.itertuples(index=False)
            )

            say(
                "[V109 WINNERS | tau={} | {}] {}"
                .format(tau,objective,choices)
            )

    for row in flip[
        flip.objective=="own_SSE"
    ].itertuples(index=False):
        say(
            "[SHIFT-SPECIFIC | {}] tau6={} | tau12={}"
            .format(
                row.culture,row.choice_6,row.choice_12
            )
        )

    for tau in DELAYS:
        x = r[
            (r.delay==tau)
            & (r.schedule!="base_0_12_24")
        ]
        say(
            "[CROSS-CULTURE | tau={}] other closer {}/16"
            .format(tau,int(x.other_closer.sum()))
        )

    say(
        "[INFERENCE BOUNDARY] Conditional templates only; "
        "no noise SD or biological delayed cultures"
    )
    return d,w

def main():
    say("="*89)
    say(
        "PAPER 7 | V109 | SHIFT-SPECIFICITY AND UNKNOWN-CULTURE DESIGN"
    )
    say(
        "MEASURED KNOTS ONLY | tau=6 vs tau=12 "
        "| NO INTERPOLATION | NO GIT"
    )
    say("="*89)

    need(len(sys.argv)==1,"Run without flags")

    run,curves,prior = load()

    protocol = dict(
        version="v109",
        source_sha256=SOURCE_SHA,
        delay_targets=DELAYS,
        observed_grid=[
            0,3,6,9,12,18,24,36,48,96
        ],
        base=BASE,
        one_extra=EXTRAS,
        score=(
            "Minimum both shift directions after "
            "gene-specific nonnegative amplitude and intercept"
        ),
        donor_unknown_score=(
            "Minimum SSE over same and three "
            "alternative culture templates"
        ),
        no_interpolation=True,
        no_delay_truth=True,
        noise_SD_known=False,
        no_git=True,
        no_network=True,
        code_sha256=digest(__file__)
    )

    (OUT/"01_protocol.json").write_text(
        json.dumps(protocol,indent=2)+"\n",
        encoding="utf-8"
    )

    distances,winners = evaluate(
        curves,prior
    )

    notes = """# V109 — Shift-specific experimental timing

V107/V108 chose the 6h additional observation
when the hypothetical target delay was exactly 6h.

V109 evaluates 6h and 12h shifted alternatives
from measured source knots without interpolation.

The source contains real expression measurements,
but neither delayed alternative was experimentally
observed.

Each gene has an unknown intercept and
nonnegative amplitude.

Own-culture design evaluates a shifted copy
of the same culture's response.

The conservative unknown-culture criterion minimizes
over four candidate templates, including the same
culture. This is not a validated classification risk.

All candidate acquisition schedules have four
timepoints; the baseline has three.

Changing the preferred measurement time when
the assumed delay changes demonstrates target-specific
design, not successful biological delay detection.

No measurement error, gene covariance or actual
experimental cost is estimated.

V104's structural ambiguity under arbitrary
post-onset shapes remains unchanged.
"""

    (OUT/"09_methods_and_limitations.md").write_text(
        notes,encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT/"script_snapshot_v109.py"
    )
    shutil.copy2(
        run/"source_measured_v67.csv",
        OUT/"input_source_v67_measured.csv"
    )
    shutil.copy2(
        run/"04_equal_cost_conditional_designs.csv",
        OUT/"input_v107_equal_cost.csv"
    )

    manifest = dict(
        version="v109",
        script_sha256=digest(__file__),
        v107_manifest_sha256=digest(
            run/"08_manifest.json"
        ),
        comparison_rows=len(distances),
        ranking_rows=len(winners),
        no_delay_truth=True,
        no_git=True,
        python=sys.version,
        numpy=np.__version__,
        pandas=pd.__version__,
        platform=platform.platform()
    )

    (OUT/"10_manifest.json").write_text(
        json.dumps(manifest,indent=2)+"\n",
        encoding="utf-8"
    )

    say(
        "[COMPLETE] 6h-vs-12h design specificity archived"
    )

def finish(ok):
    say(
        "[RESULT] "+(
            "SUCCESS" if ok else "FAILED"
        )
    )
    say("[REPORT] "+str(OUT/"console_summary.txt"))

    archive = OUT/"portable_report_v109.zip"
    say("[ZIP] "+str(archive))

    (OUT/"console_summary.txt").write_text(
        "\n".join(LOG)+"\n",encoding="utf-8"
    )

    items = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "11_sha256_inventory.csv",archive.name
        ):
            items.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(items).to_csv(
        OUT/"11_sha256_inventory.csv",
        index=False,encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p!=archive:
                z.write(p,p.name)

if __name__=="__main__":
    ok=True
    try:
        main()
    except Exception:
        ok=False
        say("[ERROR]\n"+traceback.format_exc())
    finally:
        finish(ok)
    if not ok:
        sys.exit(1)
