from pathlib import Path
from datetime import datetime, timezone
from hashlib import sha256
import json, sys, traceback, zipfile, shutil
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "paper7_early_onset_constraints_v105" / (
    "run_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
)
OUT.mkdir(parents=True, exist_ok=False)
LOG = []
SOURCE_SHA = "eae68789632bf2a6b94167d74a0efcd234e5dafa4add5d7cfcc7d1a9e42f252a"

def say(x):
    x = str(x)
    print(x, flush=True)
    LOG.append(x)

def need(ok, msg):
    if not bool(ok):
        raise RuntimeError(msg)

def digest(p):
    h = sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()

def write(name, df):
    df.to_csv(
        OUT/name, index=False,
        encoding="utf-8-sig", float_format="%.15g"
    )
    say("[OUTPUT] {} | {} rows".format(name, len(df)))

def main():
    say("="*78)
    say("PAPER 7 | V105 | REAL EARLY TIMES AND 0v6 STRUCTURAL CONSTRAINTS")
    say("NO SIMULATION | NO RE-FITTING | NO GIT")
    say("="*78)

    need(len(sys.argv) == 1, "Run without flags")

    base = ROOT / "paper7_shape_free_onset_v104"
    required = (
        "09_sha256_inventory.csv",
        "08_manifest.json",
        "01_measured_time_inventory.csv",
        "02_gene_tau_feasibility.csv",
        "05_shared_culture_tau_feasibility.csv",
        "input_V67_source.csv"
    )

    runs = sorted(
        (
            r for r in base.glob("run_*")
            if r.is_dir()
            and all((r/x).is_file() for x in required)
        ),
        reverse=True
    )
    need(runs, "V104 complete archive unavailable")
    run = runs[0]

    inv = pd.read_csv(
        run/"09_sha256_inventory.csv",
        encoding="utf-8-sig"
    )
    need(
        {"filename","bytes","sha256"}.issubset(inv.columns)
        and inv.filename.is_unique,
        "Inventory malformed"
    )

    for r in inv.itertuples(index=False):
        name = str(r.filename)
        p = run/name
        need(
            Path(name).name == name
            and p.is_file()
            and p.stat().st_size == int(r.bytes)
            and digest(p) == str(r.sha256),
            "Archived file changed: "+name
        )

    need(
        digest(run/"input_V67_source.csv") == SOURCE_SHA,
        "V67 measured source hash mismatch"
    )

    manifest = json.loads(
        (run/"08_manifest.json").read_text("utf-8")
    )
    need(
        manifest.get("version") == "v104"
        and manifest.get("n_curves") == 32
        and manifest.get("noise_estimated") is False,
        "V104 manifest mismatch"
    )

    say("[V104 VERIFIED] archived files={}".format(len(inv)))

    raw = pd.read_csv(
        run/"input_V67_source.csv",
        encoding="utf-8-sig"
    )
    feat = pd.read_csv(
        run/"02_gene_tau_feasibility.csv",
        encoding="utf-8-sig"
    )
    joint = pd.read_csv(
        run/"05_shared_culture_tau_feasibility.csv",
        encoding="utf-8-sig"
    )

    need(
        len(raw) == 320
        and len(feat) == 512
        and len(joint) == 64,
        "Unexpected archived evidence count"
    )

    need(
        not raw.duplicated(
            ["culture_label","gene","time_hours"]
        ).any(),
        "Duplicate measurements"
    )

    records = []

    for (culture,gene), d in raw.groupby(
        ["culture_label","gene"]
    ):
        d = d.sort_values("time_hours")
        t = d.time_hours.to_numpy(float)
        y = d.log2CPM.to_numpy(float)

        need(
            len(d) == 10
            and np.isclose(t,0).any()
            and not np.isclose(t,2).any(),
            "Invalid measured grid"
        )

        y0 = float(y[np.flatnonzero(np.isclose(t,0))[0]])
        pre = (t > 0) & (t <= 6)
        times = t[pre]
        shifts = y[pre]-y0
        changed = times[np.abs(shifts)>1e-9]

        results = feat[
            (feat.culture == culture)
            & (feat.gene == gene)
            & (feat.design == "all_measured")
            & (feat.tau == 6)
        ]
        need(len(results) == 2, "Missing tau6 gene evidence")

        flat = results[
            results.baseline == "flat"
        ].iloc[0]
        drift = results[
            results.baseline == "linear_drift"
        ].iloc[0]

        records.append(dict(
            culture=culture,
            gene=gene,
            measured_hours=";".join("%g"%z for z in t),
            measured_pre6=";".join("%g"%z for z in times),
            n_extra_pre6=len(times),
            max_abs_delta_from_0_log2CPM=(
                float(np.max(np.abs(shifts)))
                if len(shifts) else 0.
            ),
            earliest_nonflat_pre6=(
                float(changed[0])
                if len(changed) else np.nan
            ),
            flat_exact_noiseless=bool(flat.noiseless_exact),
            drift_exact_noiseless=bool(drift.noiseless_exact),
            drift_n_pre=int(drift.n_pre),
            drift_design_rank=int(drift.pre_rank)
        ))

    early = pd.DataFrame(records)
    write("01_measured_early_constraints.csv", early)

    summaries = []
    for (culture,design,baseline), g in joint.groupby(
        ["culture","design","baseline"]
    ):
        a = g[g.tau == 0].iloc[0]
        b = g[g.tau == 6].iloc[0]

        summaries.append(dict(
            culture=culture,
            design=design,
            baseline=baseline,
            tau0_exact=bool(a.all_8_genes_exact),
            tau6_exact=bool(b.all_8_genes_exact),
            tau0_pre_RMS=float(a.joint_pre_RMS_log2CPM),
            tau6_pre_RMS=float(b.joint_pre_RMS_log2CPM),
            tau6_n_pre_genetimes=int(b.n_pre_gene_times)
        ))

    summary = pd.DataFrame(summaries)
    write("02_core_vs_all_measured_0v6.csv", summary)

    for culture in sorted(early.culture.unique()):
        e = early[early.culture == culture]

        hours = sorted(
            set().union(*[
                set(str(v).split(";"))
                for v in e.measured_hours
            ]),
            key=float
        )

        active = int(e.earliest_nonflat_pre6.notna().sum())

        full = summary[
            (summary.culture == culture)
            & (summary.design == "all_measured")
        ]

        core = summary[
            (summary.culture == culture)
            & (summary.design == "core_measured")
        ]

        need(
            core.tau0_exact.all()
            and core.tau6_exact.all(),
            "Core 0/6 alias unexpectedly absent"
        )

        say(
            "[V105 GRID | {}] measured hours: {}"
            .format(culture,";".join(hours))
        )

        say(
            "[V105 EARLY | {}] genes changing at observed "
            "0<t<=6: {}/8 (noiseless comparison)"
            .format(culture,active)
        )

        for row in full.itertuples(index=False):
            say(
                "[V105 FULL 0v6 | {}] {} | "
                "exact_tau0={} exact_tau6={} "
                "pre_RMS_tau6={:.6f} log2CPM"
                .format(
                    culture,row.baseline,
                    row.tau0_exact,row.tau6_exact,
                    row.tau6_pre_RMS
                )
            )

    say(
        "[STRUCTURAL NESTING] Flexible C1 onset-0 "
        "can reproduce onset-6 after an internal latent phase"
    )
    say(
        "[LIMIT] Early response can contradict onset-6 "
        "under flat baseline; cannot confirm onset-6 "
        "against unrestricted onset-0"
    )
    say("[NOISE] No replicate-based sigma available")

    (OUT/"03_methods_and_limits.md").write_text(
        """# V105: exact early-onset constraints

Reuses measured source values and V104 profile fits.
No new data, simulations or classification thresholds.

With unrestricted smooth responses and a flat initial
phase, onset-0 can reproduce onset-6 by staying flat
until six hours, then following the later response.

This nesting holds for the constructed C1 smoothstep
witnesses with zero onset derivative.

An observed change before or at six hours may contradict
the onset-6 model under an exact flat-baseline assumption.
No statistical exclusion is possible without measured noise.

Free drift can absorb early variation. If fewer than
two independent pre-onset times exist, drift is not
identified; inspect the reported design rank.

No universal delay identifiability is claimed under
unknown response shape. Real replicate SD is absent
from the frozen analysis snapshot.
""",
        encoding="utf-8"
    )

    shutil.copy2(
        Path(__file__),
        OUT/"script_snapshot_v105.py"
    )
    shutil.copy2(
        run/"08_manifest.json",
        OUT/"input_v104_manifest.json"
    )

    (OUT/"04_manifest.json").write_text(
        json.dumps(dict(
            version="v105",
            source_sha256=SOURCE_SHA,
            script_sha256=digest(__file__),
            v104_run=str(run),
            no_new_simulation=True,
            noise_sigma_available=False,
            no_git=True
        ),indent=2),
        encoding="utf-8"
    )

    say("[COMPLETE] Early measured-time audit archived")

def finish(ok):
    say("[RESULT] "+("SUCCESS" if ok else "FAILED"))
    say("[REPORT] "+str(OUT/"console_summary.txt"))
    archive = OUT/"portable_report_v105.zip"
    say("[ZIP] "+str(archive))

    (OUT/"console_summary.txt").write_text(
        "\n".join(LOG)+"\n",
        encoding="utf-8"
    )

    entries = []
    for p in sorted(OUT.iterdir()):
        if p.is_file() and p.name not in (
            "05_sha256_inventory.csv",archive.name
        ):
            entries.append(dict(
                filename=p.name,
                bytes=p.stat().st_size,
                sha256=digest(p)
            ))

    pd.DataFrame(entries).to_csv(
        OUT/"05_sha256_inventory.csv",
        index=False,encoding="utf-8-sig"
    )

    with zipfile.ZipFile(
        archive,"w",zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(OUT.iterdir()):
            if p.is_file() and p != archive:
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
