from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json, sys, traceback, zipfile, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_shape_free_onset_v104" / (
    "run_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)

LOG = []
GENES = ("PRL","IGFBP1","FOXO1","HAND2","WNT4","PGR","IL11","LEFTY2")
CULTURES = ("S169","S170","S506","S508")
TAUS = (0., 6., 12., 24.)
CORE = (0., 12., 24., 36., 48.)
SHA_SOURCE = "eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a"

def say(s):
    s = str(s)
    print(s, flush=True)
    LOG.append(s)

def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def sha(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()

def save(name, rows):
    df = pd.DataFrame(rows)
    df.to_csv(
        OUT / name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | rows={}".format(name, len(df)))
    return df

def archive():
    base = ROOT / "paper7_measured_identifiability_v103"
    runs = sorted(
        (p for p in base.glob("run_*")
         if (p / "input_v67_measured_source.csv").is_file()
         and (p / "11_manifest.json").is_file()
         and (p / "12_sha256_inventory.csv").is_file()),
        reverse=True
    )

    for run in runs:
        path = run / "input_v67_measured_source.csv"
        if sha(path) != SHA_SOURCE:
            continue

        inv = pd.read_csv(
            run / "12_sha256_inventory.csv",
            encoding="utf-8-sig"
        )
        need(
            {"filename","bytes","sha256"}.issubset(inv.columns),
            "V103 inventory malformed"
        )

        for r in inv.itertuples(index=False):
            fn = str(r.filename)
            f = run / fn
            need(
                Path(fn).name == fn
                and f.is_file()
                and f.stat().st_size == int(r.bytes)
                and sha(f) == str(r.sha256),
                "V103 archived input changed: " + fn
            )

        m = json.loads(
            (run / "11_manifest.json").read_text("utf-8")
        )
        need(
            m.get("version") == "v103"
            and m.get("n_curves") == 32
            and m.get("noise_sd_available") is False,
            "V103 manifest mismatch"
        )

        say("[V103 VERIFIED] source SHA256=" + SHA_SOURCE)
        return run, path

    raise RuntimeError("Verified V103 source not found")

def c1_response(x, knots, values):
    # Smooth cubic interpolation, with zero derivative at knots.
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    v = np.asarray(values, float)

    if len(k) == 1:
        return np.zeros_like(x)

    i = np.clip(
        np.searchsorted(k, x, side="right") - 1,
        0, len(k)-2
    )
    u = np.clip((x-k[i])/(k[i+1]-k[i]), 0, 1)
    z = v[i] + (v[i+1]-v[i])*(3*u*u-2*u*u*u)
    return np.where(x <= 0, 0, np.where(x >= k[-1], v[-1], z))

def run():
    run103, src = archive()
    raw = pd.read_csv(src, encoding="utf-8-sig")

    need(
        {"culture_label","gene","time_hours","log2CPM"}.issubset(raw.columns),
        "Source schema changed"
    )
    raw = raw[
        raw.culture_label.isin(CULTURES)
        & raw.gene.isin(GENES)
    ]
    need(
        len(raw) == 320
        and not raw.duplicated(
            ["culture_label","gene","time_hours"]
        ).any(),
        "Expected 4 x 8 x 10 measured values"
    )

    protocol = {
        "version": "v104",
        "input_sha256": sha(src),
        "delays": TAUS,
        "baseline_models": ["flat","linear_drift"],
        "response_class": "arbitrary C1 q(s), q(0)=0",
        "noise_sd_available": False,
        "biological_delay_truth_available": False,
        "no_git": True,
        "no_network": True,
        "script_sha256": sha(__file__)
    }
    (OUT / "00_protocol.json").write_text(
        json.dumps(protocol, indent=2) + "\n",
        encoding="utf-8"
    )

    inventory, results, witnesses, recon = [], [], [], []

    for cult in CULTURES:
        for gene in GENES:
            block = raw[
                (raw.culture_label == cult)
                & (raw.gene == gene)
            ].sort_values("time_hours")

            need(len(block) == 10, "Incomplete source curve")
            t = block.time_hours.to_numpy(float)
            y = block.log2CPM.to_numpy(float)

            need(
                np.isfinite(t).all()
                and np.isfinite(y).all()
                and np.all(np.diff(t) > 0),
                "Invalid measurements"
            )
            need(
                np.isclose(t, 0).any()
                and np.isclose(t, 36).any()
                and not np.isclose(t, 2).any(),
                "Unexpected measured times"
            )

            core = np.asarray(
                [x for x in CORE if np.isclose(t, x).any()],
                float
            )
            need(len(core) >= 3, "Not enough core times")

            inventory.append(dict(
                culture=cult, gene=gene,
                measured_hours=";".join("%g" % x for x in t),
                n_measured=len(t),
                at_2h=False, at_36h=True
            ))

            for design, grid in (
                ("all_measured", t),
                ("core_measured", core)
            ):
                yy = np.asarray([
                    y[np.argmin(abs(t-x))] for x in grid
                ])

                for tau in TAUS:
                    pre = grid <= tau + 1e-10
                    post = ~pre

                    for kind in ("flat","linear_drift"):
                        if kind == "linear_drift":
                            X = np.column_stack((
                                np.ones(pre.sum()), grid[pre]
                            ))
                        else:
                            X = np.ones((pre.sum(),1))

                        coef = np.linalg.lstsq(
                            X, yy[pre], rcond=None
                        )[0]

                        b = float(coef[0])
                        d = (
                            float(coef[1])
                            if kind == "linear_drift" else 0.
                        )

                        residual = yy[pre] - (
                            b + d*grid[pre]
                        )
                        sse = float(residual @ residual)

                        knots = np.r_[0., grid[post]-tau]
                        values = np.r_[
                            0., yy[post]-b-d*grid[post]
                        ]

                        fitted = (
                            b + d*grid
                            + c1_response(grid-tau, knots, values)
                        )

                        if post.any():
                            need(
                                np.max(abs(
                                    yy[post]-fitted[post]
                                )) < 1e-10,
                                "Post-onset C1 witness failed"
                            )
                        need(
                            abs(
                                float(np.sum((yy-fitted)**2))
                                - sse
                            ) < 1e-9,
                            "Pre-onset SSE mismatch"
                        )

                        results.append(dict(
                            culture=cult,
                            gene=gene,
                            design=design,
                            tau=tau,
                            baseline=kind,
                            n_pre=int(pre.sum()),
                            n_post=int(post.sum()),
                            pre_rank=int(np.linalg.matrix_rank(X)),
                            pre_RMS_log2CPM=float(
                                np.sqrt(sse/pre.sum())
                            ),
                            pre_SSE=sse,
                            noiseless_exact=bool(
                                np.max(abs(residual)) <= 1e-9
                            ),
                            noise_SD_known=False
                        ))

                        for ix, (tt, vv) in enumerate(
                            zip(knots, values)
                        ):
                            witnesses.append(dict(
                                culture=cult,
                                gene=gene,
                                design=design,
                                tau=tau,
                                baseline=kind,
                                knot=ix,
                                after_onset_hours=tt,
                                response_log2CPM=vv
                            ))

                        for hour, actual, fit in zip(
                            grid, yy, fitted
                        ):
                            recon.append(dict(
                                culture=cult,
                                gene=gene,
                                design=design,
                                tau=tau,
                                baseline=kind,
                                time_hours=hour,
                                observed_log2CPM=actual,
                                fitted_log2CPM=fit,
                                residual_log2CPM=actual-fit,
                                pre_onset=bool(hour <= tau+1e-10)
                            ))

        say("[MEASURED] {} | 8 genes x 10 times".format(cult))

    save("01_measured_time_inventory.csv", inventory)
    df = save("02_gene_tau_feasibility.csv", results)
    save("03_C1_witness_knots.csv", witnesses)
    save("04_reconstructed_measured_knots.csv", recon)

    shared, pairs = [], []

    for (cult, design, kind, tau), group in df.groupby(
        ["culture","design","baseline","tau"]
    ):
        need(len(group) == 8, "Missing panel genes")
        sse = group.pre_SSE.sum()
        n = group.n_pre.sum()

        shared.append(dict(
            culture=cult, design=design,
            baseline=kind, tau=tau,
            joint_pre_RMS_log2CPM=float(np.sqrt(sse/n)),
            n_pre_gene_times=int(n),
            all_8_genes_exact=bool(
                group.noiseless_exact.all()
            ),
            empirical_noise_SD_available=False
        ))

    shared = save(
        "05_shared_culture_tau_feasibility.csv", shared
    )

    for (cult, design, kind), group in shared.groupby(
        ["culture","design","baseline"]
    ):
        g = group.set_index("tau")

        for a,b in combinations(TAUS,2):
            pairs.append(dict(
                culture=cult,
                design=design,
                baseline=kind,
                tau_a=a,
                tau_b=b,
                both_fit_exactly=bool(
                    g.loc[a,"all_8_genes_exact"]
                    and g.loc[b,"all_8_genes_exact"]
                ),
                tau_a_pre_RMS=g.loc[a,"joint_pre_RMS_log2CPM"],
                tau_b_pre_RMS=g.loc[b,"joint_pre_RMS_log2CPM"]
            ))

    save("06_pairwise_shared_compatibility.csv", pairs)

    previous = pd.read_csv(
        run103 / "07_culture_shared_delay_pairwise.csv",
        encoding="utf-8-sig"
    )
    old = previous[
        (previous.schedule == "core_measured")
        & (previous.tau_a == 0)
        & (previous.tau_b == 6)
        & (previous.nuisance == "baseline_amplitude")
    ]
    need(len(old) == 4, "Missing V103 comparison")

    for cult in CULTURES:
        flat = shared[
            (shared.culture == cult)
            & (shared.design == "core_measured")
            & (shared.baseline == "flat")
            & (shared.tau == 6)
        ].iloc[0]

        drift = shared[
            (shared.culture == cult)
            & (shared.design == "core_measured")
            & (shared.baseline == "linear_drift")
            & (shared.tau == 6)
        ].iloc[0]

        pchip = old[old.culture == cult].iloc[0]

        say(
            "[V104 0v6 | {}] flat-pre RMS={:.5f} "
            "| drift-pre RMS={:.5f} "
            "| V103 conditional PCHIP={:.5f} log2CPM"
            .format(
                cult,
                flat.joint_pre_RMS_log2CPM,
                drift.joint_pre_RMS_log2CPM,
                pchip.joint_RMS_log2CPM
            )
        )

    say(
        "[STRUCTURAL] All measured post-onset knots "
        "fit exactly with arbitrary C1 responses"
    )
    say(
        "[NOISE GATE] Replicate SD unavailable; "
        "practical detectability NOT EVALUABLE"
    )

    (OUT / "07_methods_limitations.md").write_text(
        """# Paper 7 V104: shape-free onset audit

Measured source only: 32 gene/culture trajectories,
10 timepoints per trajectory.

Model: y(t) = b + d*t + q(max(t-tau,0)).
q(0)=0; arbitrary continuously differentiable
post-onset response.

Two baseline classes: flat, and free linear drift.

A cubic C1 smoothstep witness reproduces every
post-onset measured point exactly.
Remaining residuals are entirely pre-onset.

Unknown gene-specific baseline and drift are
fitted separately; tau is common across eight
genes within a culture.

No replicate-derived noise SD and no observed
biological delay labels are available.

The response family permits arbitrary shapes
and is not claimed to be biologically realistic.

V103 PCHIP distances impose additional restrictions
and are not the same estimand as V104 pre-onset RMS.

The source has no measured 2h values.
No imputation or new synthetic data are used.
""",
        encoding="utf-8"
    )

    shutil.copy2(src, OUT / "input_V67_source.csv")
    shutil.copy2(
        Path(__file__), OUT / "script_snapshot_v104.py"
    )

    (OUT / "08_manifest.json").write_text(
        json.dumps(dict(
            version="v104",
            source_SHA256=SHA_SOURCE,
            script_sha256=sha(__file__),
            noise_estimated=False,
            practical_identifiability="NOT EVALUABLE",
            n_curves=32,
            no_git=True
        ), indent=2),
        encoding="utf-8"
    )

    say("[COMPLETE] Shape-free feasibility and witnesses archived")

def finish(success):
    say("[RESULT] " + ("SUCCESS" if success else "FAILED"))
    say("[REPORT] " + str(OUT / "console_summary.txt"))
    archive = OUT / "portable_report_v104.zip"
    say("[ZIP] " + str(archive))

    (OUT / "console_summary.txt").write_text(
        "\n".join(LOG) + "\n",
        encoding="utf-8"
    )

    hashes = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            archive.name, "09_sha256_inventory.csv"
        ):
            hashes.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=sha(p)
            ))

    pd.DataFrame(hashes).to_csv(
        OUT / "09_sha256_inventory.csv",
        index=False,
        encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive, "w", zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
                z.write(p, p.name)

if __name__ == "__main__":
    success = True
    try:
        run()
    except Exception:
        success = False
        say("[ERROR]\n" + traceback.format_exc())
    finally:
        finish(success)
    if not success:
        sys.exit(1)
