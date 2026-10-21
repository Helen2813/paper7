#!/usr/bin/env python3
"""Paper 7, phase-0: GEO metadata audit and CONTROL-ONLY timing/strength identifiability.

This is an exploratory feasibility tool, not a clinical classifier or validated model.
No case expression is used, even if cases exist in the downloaded matrices.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import itertools
import json
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import numpy as np
import pandas as pd
import requests
from scipy.interpolate import PchipInterpolator, UnivariateSpline
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = ["GSE243039", "GSE246591", "GSE35287", "GSE145701", "GSE130028"]
EXPECTED_SAMPLES = {"GSE243039": 40, "GSE246591": 40, "GSE35287": 80, "GSE145701": 48, "GSE130028": 48}
SUPPLEMENT = {
    "GSE243039": "GSE243039_normalized_all_counts.csv.gz",
    "GSE246591": "GSE246591_Decidualization_timecourse_RNAseq_counts_matrix.txt.gz",
}
EXPRESSION_KIND = {"GSE243039": "normalized_counts", "GSE246591": "raw_counts"}
USER_AGENT = "Paper7Pilot/0.1 (+NCBI GEO research; no automated bulk SRA)"


def stub(gse: str) -> str:
    return re.sub(r"\d{1,3}$", "nnn", gse)


def series_dir(gse: str, subfolder: str) -> str:
    return f"https://ftp.ncbi.nlm.nih.gov/geo/series/{stub(gse)}/{gse}/{subfolder}/"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for piece in iter(lambda: fh.read(1 << 20), b""):
            h.update(piece)
    return h.hexdigest()


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def http_download(session: requests.Session, url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".partial")
    for attempt in range(3):
        try:
            with session.get(url, timeout=(20, 120), stream=True) as response:
                response.raise_for_status()
                with tmp.open("wb") as out:
                    for chunk in response.iter_content(chunk_size=1 << 18):
                        if chunk:
                            out.write(chunk)
            tmp.replace(destination)
            return destination
        except requests.RequestException:
            if attempt == 2:
                raise
    raise AssertionError("Unreachable")


def parse_matrix_headers(lines: list[str], gse: str) -> pd.DataFrame:
    """Parse GEO SERIES MATRIX header-only lines, including repeated characteristics."""
    keys: dict[str, list[list[str]]] = {}
    for line in lines:
        if not line.startswith("!Sample_"):
            continue
        rec = next(csv.reader([line.rstrip("\n\r")], delimiter="\t"))
        if len(rec) < 2:
            continue
        key = rec[0].removeprefix("!Sample_")
        keys.setdefault(key, []).append(rec[1:])
    if "geo_accession" not in keys:
        raise ValueError(f"{gse}: !Sample_geo_accession missing; verify Series Matrix download")
    gsm_ids = keys["geo_accession"][0]
    n = len(gsm_ids)
    data: dict[str, list[str]] = {"gse": [gse] * n, "gsm": gsm_ids}
    for key, repeated_rows in keys.items():
        if key == "geo_accession":
            continue
        if any(len(row) != n for row in repeated_rows):
            raise ValueError(f"{gse}: inconsistent !Sample_{key} field length")
        data[key] = [" | ".join(row[i] for row in repeated_rows if row[i]).strip() for i in range(n)]
    df = pd.DataFrame(data)
    if df.gsm.duplicated().any():
        raise ValueError(f"{gse}: repeated GSM identifiers")
    return df


def get_matrix_header(session: requests.Session, gse: str, output: Path, offline: bool) -> pd.DataFrame:
    cache = output / "metadata" / f"{gse}_series_header.txt"
    if cache.exists() and cache.stat().st_size:
        return parse_matrix_headers(cache.read_text(encoding="utf-8").splitlines(), gse)
    if offline:
        raise FileNotFoundError(f"Missing {cache}; run without --offline on a computer with Internet")
    candidates = [urljoin(series_dir(gse, "matrix"), f"{gse}_series_matrix.txt.gz")]
    # Some GEO records use platform-specific matrix filenames. Look for those if needed.
    try:
        res = session.get(series_dir(gse, "matrix"), timeout=30)
        if res.ok:
            found = re.findall(r'href=["\']([^"\']*series_matrix\.txt\.gz)["\']', res.text, re.I)
            candidates = list(dict.fromkeys(candidates + [urljoin(series_dir(gse, "matrix"), s) for s in found]))
    except requests.RequestException:
        pass
    errors = []
    for url in candidates:
        try:
            with session.get(url, timeout=(20, 120), stream=True) as response:
                response.raise_for_status()
                response.raw.decode_content = False
                with gzip.GzipFile(fileobj=response.raw) as gz:
                    raw_lines = []
                    for binary_line in gz:
                        line = binary_line.decode("utf-8", errors="replace")
                        if line.startswith("!series_matrix_table_begin"):
                            break  # crucial: avoid downloading enormous methylation matrices
                        raw_lines.append(line)
                        if sum(map(len, raw_lines)) > 15_000_000:
                            raise RuntimeError(f"{gse} unexpectedly large header")
            metadata = parse_matrix_headers(raw_lines, gse)
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text("".join(raw_lines), encoding="utf-8")
            return metadata
        except (requests.RequestException, OSError, EOFError, ValueError) as e:
            errors.append(f"{url}: {e}")
    raise RuntimeError(f"{gse}: could not retrieve GEO matrix header:\n" + "\n".join(errors))


def explicit_time(title: str, char: str, gse: str) -> float:
    if gse in {"GSE145701", "GSE130028"}:
        return 336.0  # nominal 14-day endpoint; from GEO Overall Design, not per-sample time series
    # Title first. Do NOT extract protocol overview times from generic descriptive paragraphs.
    for text in [title, " | ".join(x for x in char.split(" | ") if re.search(r"\btime\s*:", x, re.I))]:
        m = re.search(r"\bday\s*([0-9]+)\b", text, re.I)
        if m:
            return 24.0 * float(m.group(1))
        m = re.search(r"(?<!\w)(\d+(?:\.\d+)?)\s*h(?:r|our)?s?\b", text, re.I)
        if m:
            return float(m.group(1))
        m = re.search(r"(?:^|\s)D([0-9]+)(?:\s|$)", text, re.I)
        if m:
            return 24.0 * float(m.group(1))
    return float("nan")


def infer_annotation(rec: pd.Series) -> dict[str, object]:
    gse = str(rec.gse)
    title = str(rec.get("title", ""))
    characteristic = str(rec.get("characteristics_ch1", ""))
    combined = title + " | " + characteristic
    low = combined.lower()
    group = "unknown"
    local_donor = ""
    treatment = "unknown"
    if gse == "GSE243039":
        if "normal endometrial" in low:
            group = "control"
        elif "endometriosis stroma" in low:
            group = "endometriosis"
        m = re.search(r"replicate\s*(\d+)", title, re.I)
        if m:
            local_donor = f"{group}:rep{m.group(1)}"
        treatment = "EPC"  # estrogen, progestin, cAMP
    elif gse == "GSE246591":
        m = re.search(r"\b(S\d+)\s+\d+\s*h\b", title, re.I)
        if m:
            local_donor = m.group(1).upper()
        # GEO public description does not confirm disease-free controls
        group = "reference_status_unverified"
        treatment = "cAMP+MPA"
    elif gse in {"GSE145701", "GSE130028"}:
        if "nup" in low:
            group = "control"
        elif re.search(r"endo\s*(?:iv|4)\b", low):
            group = "endo_stage_iv"
        elif re.search(r"endo\s*(?:i|1)\b", low):
            group = "endo_stage_i"
        m = re.search(r"sample\s*(\d+)", title, re.I)
        if m:
            local_donor = f"{group}:sample{m.group(1)}"
        if re.search(r"\bE2\s*\+\s*P4\b", title, re.I):
            treatment = "E2+P4"
        elif re.search(r"\btreated with\s*E2\b", title, re.I):
            treatment = "E2"
        elif re.search(r"\btreated with\s*P4\b", title, re.I):
            treatment = "P4"
        elif re.search(r"\b(?:vehicle|veh)\b", title, re.I):
            treatment = "vehicle"
    elif gse == "GSE35287":
        if "nonendo" in low or "without endometriosis" in low or "non-endometriosis" in low:
            group = "control"
        elif "endometriosis" in low or "hESFendo".lower() in low:
            group = "endometriosis"
        m = re.search(r"(?:patient|donor|subject|biological replicate)\s*(?:id|no\.)?\s*[:#-]?\s*(\d+)", combined, re.I)
        if m:
            local_donor = f"{group}:subject{m.group(1)}"
        if re.search(r"\bP4\s*\+\s*E2\b|\bE2\s*\+\s*P4\b", title, re.I):
            treatment = "E2+P4"
        elif re.search(r"\bE2\b", title, re.I):
            treatment = "E2"
        elif "vehicle" in low:
            treatment = "vehicle"
    return {"group": group, "donor_local": local_donor, "time_hours": explicit_time(title, characteristic, gse), "treatment_pilot": treatment}


def metadata_audit(out: Path, session: requests.Session, offline: bool) -> pd.DataFrame:
    frames, reports = [], []
    errors = []
    for gse in SERIES:
        try:
            df = get_matrix_header(session, gse, out, offline)
            ann = df.apply(infer_annotation, axis=1, result_type="expand")
            df = pd.concat([df, ann], axis=1)
            if len(df) != EXPECTED_SAMPLES[gse]:
                errors.append(f"{gse}: got {len(df)} samples, GEO reports {EXPECTED_SAMPLES[gse]}")
            df.to_csv(out / "metadata" / f"{gse}_samples.csv", index=False)
            frames.append(df)
            reports.append({"gse": gse, "n_samples": len(df), "n_donors_parsed": int(df.donor_local[df.donor_local != ""].nunique()),
                            "n_donor_unknown_samples": int((df.donor_local == "").sum()), "time_points_hours": ";".join(map(str, sorted(df.time_hours.dropna().unique()))),
                            "groups": ";".join(f"{k}:{v}" for k, v in df.group.value_counts().items()),
                            "unknown_group": int((df.group == "unknown").sum())})
            print(f"[GEO] {gse}: {len(df)} samples; donor labels for {(df.donor_local!='').sum()} samples; times={sorted(df.time_hours.dropna().unique())}")
        except Exception as e:
            errors.append(f"{gse}: {type(e).__name__}: {e}")
            print(f"[WARNING] {gse} metadata unavailable: {e}", file=sys.stderr)
    pd.DataFrame(reports).to_csv(out / "metadata" / "study_summary.csv", index=False)
    overlap = []
    pair_data = {f.gse.iloc[0]: f for f in frames if not f.empty}
    for a, b in itertools.combinations(SERIES, 2):
        matched_keys = ""
        n_matching = ""
        if (a, b) == ("GSE145701", "GSE130028"):
            status = "paired expression/methylation designs in SuperSeries GSE145702"
            if a in pair_data and b in pair_data:
                left = pair_data[a]
                right = pair_data[b]
                fields = ["group", "donor_local", "treatment_pilot"]
                left_keys = set(map(tuple, left.loc[left.donor_local.ne(""), fields].itertuples(index=False, name=None)))
                right_keys = set(map(tuple, right.loc[right.donor_local.ne(""), fields].itertuples(index=False, name=None)))
                shared = left_keys & right_keys
                n_matching = len(shared)
                matched_keys = ";".join("/".join(k) for k in sorted(shared))
                if len(shared) != 48:
                    status += "; ATTENTION: not all 48 donor×treatment keys match automatically; manually check"
                else:
                    status += "; all 48 donor×treatment labels correspond (sample identity still requires provenance review)"
        elif a == "GSE35287" and b in {"GSE145701", "GSE130028"}:
            status = "Giudice-related laboratories; donor re-use not documented, DO NOT ASSUME"
        else:
            status = "NO VERIFIED CROSS-SERIES DONOR LINK; same local ID never proves identity"
        overlap.append({"gse_1": a, "gse_2": b, "n_matching_paired_design_labels": n_matching,
                        "paired_design_shared_keys": matched_keys, "status": status})
    pd.DataFrame(overlap).to_csv(out / "metadata" / "cross_study_donor_status.csv", index=False)
    (out / "metadata" / "audit_warnings.txt").write_text("\n".join(errors) + "\n" +
        "GSE246591 disease-free control status is not established by sample titles: treat as reference-status-unverified.\n"
        "GSE35287 donor/time annotations may require manual curation. Its GEO entry reports 80 samples.\n"
        "GSE145701 and GSE130028 are paired RNA/methylation experiments, not independent donor cohorts.\n"
        "Do not count repeated timepoints or conditions as independent donors.\n", encoding="utf-8")
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_matrix(path: Path) -> pd.DataFrame:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        header = fh.readline()
    separator = "\t" if "\t" in header else ","
    matrix = pd.read_csv(path, sep=separator, compression="infer", low_memory=False)
    if matrix.shape[1] < 3:
        raise ValueError(f"{path.name}: expression matrix has fewer than 2 sample columns")
    return matrix


def locate_expression(out: Path, session: requests.Session, gse: str, offline: bool) -> Path:
    filename = SUPPLEMENT[gse]
    path = out / "raw" / gse / filename
    if path.exists() and path.stat().st_size:
        return path
    if offline:
        raise FileNotFoundError(f"Missing {path}; download supplementary file manually or run without --offline")
    url = urljoin(series_dir(gse, "suppl"), filename)
    print(f"[Download] {url}")
    return http_download(session, url, path)


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def match_columns(expr_names: list[str], metadata: pd.DataFrame, gse: str, out: Path) -> dict[str, str]:
    """Conservative mapping. Never guess ambiguous sample order."""
    manual_path = out / "metadata" / "sample_column_mapping.csv"
    manual = pd.read_csv(manual_path).fillna("") if manual_path.exists() else pd.DataFrame()
    if not manual.empty and not {"gse", "expression_column", "gsm"}.issubset(manual):
        raise ValueError(f"{manual_path} needs gse,expression_column,gsm columns")
    overrides = {}
    if not manual.empty:
        overrides = dict(zip(manual.loc[manual.gse == gse, "expression_column"], manual.loc[manual.gse == gse, "gsm"]))
    mapping: dict[str, str] = {}
    meta_records = metadata.to_dict("records")
    for col in expr_names:
        gsm = overrides.get(col)
        if gsm:
            if gsm not in set(metadata.gsm):
                raise ValueError(f"{gse}: manual map {col}->{gsm} not in GEO")
            mapping[col] = gsm
            continue
        name = normalize_name(col)
        candidates = []
        for r in meta_records:
            if name in {normalize_name(r["gsm"]), normalize_name(r.get("title", ""))}:
                candidates.append(r["gsm"])
        if len(candidates) != 1 and gse == "GSE246591":
            # Explicitly restrict fallback to unique S###_<hours> names in GSE246591.
            for r in meta_records:
                donor, t = str(r.get("donor_local", "")), r.get("time_hours")
                if donor and pd.notna(t):
                    tokens = [f"{donor}_{int(t)}h", f"{donor}.{int(t)}h", f"{donor} {int(t)}h"]
                    if name in {normalize_name(z) for z in tokens}:
                        candidates.append(r["gsm"])
        candidates = sorted(set(candidates))
        if len(candidates) == 1:
            mapping[col] = candidates[0]
    proposed_path = out / "metadata" / f"{gse}_expression_column_candidates.csv"
    pd.DataFrame([{"gse":gse, "expression_column":s, "mapped_gsm": mapping.get(s, "")} for s in expr_names]).to_csv(proposed_path, index=False)
    if len(mapping) != len(expr_names) or len(set(mapping.values())) != len(mapping):
        raise ValueError(f"{gse}: mapped {len(mapping)}/{len(expr_names)} columns with unambiguous IDs. "
                         f"Inspect {proposed_path} and add explicit rows to {manual_path} "
                         "(columns gse,expression_column,gsm). No mapping by column order is allowed.")
    return mapping


def gene_symbols_and_expression(table: pd.DataFrame, out: Path, gse: str) -> tuple[pd.DataFrame, bool]:
    idx_name = str(table.columns[0])
    first = table.iloc[:, 0].astype(str).str.strip()
    # Use first-column gene symbols, or a dedicated symbol column. Do not pretend ENSG IDs are symbols.
    symbolic = None
    sym_col = next((c for c in table.columns[:4] if re.fullmatch(r"(?:gene[_ .]?symbol|symbol|hgnc[_ .]?symbol)", str(c), re.I)), None)
    if sym_col is not None:
        symbolic = table[sym_col].astype(str).str.strip()
        table = table.drop(columns=[sym_col])
        if sym_col == idx_name:
            idx_name = str(table.columns[0])
        elif str(table.columns[0]).lower() in {"geneid", "gene_id", "id"}:
            table = table.iloc[:, 1:]
    else:
        symbolic = first
        table = table.iloc[:, 1:]
    if symbolic.str.contains(r"^ENSG\d+", regex=True).mean() > 0.5:
        mapping_path = out / "metadata" / "gene_id_to_symbol.csv"
        if not mapping_path.exists():
            pd.DataFrame({"gene_id": symbolic.head(25), "gene_symbol": [""]*min(len(symbolic),25)}).to_csv(
                out / "metadata" / "gene_id_to_symbol_TEMPLATE.csv", index=False)
            raise ValueError(f"{gse}: expression contains Ensembl IDs rather than gene symbols. "
                             f"Provide checked mapping file {mapping_path} with gene_id,gene_symbol. "
                             "Never silently treat ENSG as HGNC symbols.")
        gm = pd.read_csv(mapping_path).fillna("")
        mapping = dict(zip(gm.gene_id.astype(str).str.split(".", regex=False).str[0], gm.gene_symbol.astype(str)))
        symbolic = symbolic.str.split(".", regex=False).str[0].map(mapping)
    usable = symbolic.notna() & symbolic.astype(str).str.fullmatch(r"[A-Za-z][A-Za-z0-9.-]{1,29}")
    if usable.sum() < 20:
        raise ValueError(f"{gse}: only {usable.sum()} symbol-like IDs; check column layout and gene mapping")
    table.index = symbolic.astype(str).str.upper().str.strip()
    table = table.loc[usable.values]
    table = table.apply(pd.to_numeric, errors="coerce").groupby(level=0).mean()
    table = table.dropna(axis=0, how="all")
    return table, True


def prepare_scores(expression: pd.DataFrame, metadata: pd.DataFrame, programs: dict[str, list[str]], kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if kind == "raw_counts":
        # Within dataset only, library-size normalized. Do not compare raw log counts cross-protocol.
        x = expression.clip(lower=0)
        lib = x.sum(axis=0).replace(0, np.nan)
        logx = np.log2(x.div(lib, axis=1) * 1e6 + 1)
    else:
        logx = np.log2(expression.clip(lower=0) + 1)
    assert logx.shape[1] == len(metadata) and (list(logx.columns) == list(metadata.gsm)), "Sample alignment error"
    # Only control in GSE243039; GSE246591 is an UNVERIFIED reference until donor health status is documented.
    group = metadata.group.iloc[0]
    if not set(metadata.group.unique()) <= {"control", "reference_status_unverified"}:
        raise AssertionError("Case expression was passed to control-only module scoring")
    valid = logx.loc[logx.notna().sum(axis=1) >= max(3, int(0.8 * logx.shape[1]))]
    means = valid.mean(axis=1)
    std = valid.std(axis=1).replace(0, np.nan)
    z = valid.sub(means, axis=0).div(std, axis=0)
    records, coverage = [], []
    for name, genes in programs.items():
        present = sorted(set(g.upper() for g in genes) & set(z.index))
        coverage.append({"program": name, "n_requested": len(genes), "n_present": len(present), "present_genes": ";".join(present)})
        if len(present) < 3:
            continue
        s = z.loc[present].mean(axis=0)
        for gsm, score in s.items():
            records.append({"gsm": gsm, "program": name, "score": float(score)})
    return pd.DataFrame(records), pd.DataFrame(coverage)


def read_gmt(path: Path) -> dict[str, list[str]]:
    programs: dict[str, list[str]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            cells = line.strip().split("\t")
            if len(cells) < 5:
                raise ValueError("GMT format name TAB description TAB gene1 TAB gene2 TAB gene3...")
            programs[cells[0]] = list(dict.fromkeys(cells[2:]))
    return programs


def identifiability_diagnostics(time: np.ndarray, mean: np.ndarray, smooth: bool = True) -> dict[str, float]:
    """Local identifiability of intercept + amplitude*f(t) + delay*(-f'(t)), not full nonlinear proof."""
    order = np.argsort(time)
    x = np.asarray(time, dtype=float)[order]
    y = np.asarray(mean, dtype=float)[order]
    if len(x) < 4 or len(set(x)) < 4:
        raise ValueError("Require >=4 distinct time points")
    if smooth and len(x) >= 5:
        # Small regularization, heuristic. Interpretation requires smoothing sensitivity analysis.
        interpolator = UnivariateSpline(x, y, k=3, s=0.03*len(x))
        dy = interpolator.derivative()(x)
    else:
        interpolator = PchipInterpolator(x, y)
        dy = interpolator.derivative()(x)
    # Remove intercept; normalize columns since amplitude and time have different physical units.
    u = y - np.mean(y)
    v = dy - np.mean(dy)
    su = np.linalg.norm(u)
    sv = np.linalg.norm(v)
    if min(su, sv) < 1e-9:
        return {"abs_corr_f_fprime": np.nan, "condition_number": np.inf, "det_norm": 0.0,
                "rank_two": False, "norm_f": float(su), "norm_fprime": float(sv)}
    X = np.column_stack([u/su, v/sv])
    sing = np.linalg.svd(X, compute_uv=False)
    corr = float(np.dot(X[:, 0], X[:, 1]))
    return {"abs_corr_f_fprime": abs(corr), "condition_number": float(sing.max()/sing.min()) if sing.min()>1e-10 else np.inf,
            "det_norm": float(np.linalg.det(X.T @ X)), "rank_two": bool(sing.min()>1e-8),
            "norm_f": float(su), "norm_fprime": float(sv)}


def plot_and_diagnose(scores: pd.DataFrame, out: Path, gse: str) -> pd.DataFrame:
    outputs = []
    if scores.empty:
        return pd.DataFrame()
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for prog, df in scores.groupby("program"):
        df = df.dropna(subset=["score", "time_hours"]).copy()
        if df.empty:
            continue
        df["days"] = df.time_hours / 24.0
        plot = df.pivot_table(index="days", columns="donor_local", values="score", aggfunc="mean")
        if plot.shape[0] < 4:
            continue
        t = plot.index.to_numpy(dtype=float)
        ys = plot.mean(axis=1).to_numpy(dtype=float)
        row = {"gse": gse, "program": prog, "n_timepoints": len(t), "n_donors": plot.shape[1],
               "min_time_days": float(t.min()), "max_time_days": float(t.max())}
        for setting in [False, True]:
            try:
                res = identifiability_diagnostics(t, ys, smooth=setting)
                suffix = "smooth" if setting else "pchip"
                for key, val in res.items():
                    row[f"{suffix}_{key}"] = val
            except ValueError as e:
                row["error"] = str(e)
        outputs.append(row)
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        for donor in plot.columns:
            ax.plot(t, plot[donor].to_numpy(dtype=float), "o-", alpha=0.35, linewidth=1, markersize=4)
        ax.plot(t, ys, "o-", linewidth=2.3, color="black", markersize=5, label="donor mean")
        ax.set(xlabel="Time after treatment (days)", ylabel="Within-study control gene-program z score",
               title=f"{gse}: {prog} (CONTROL/REFERENCE ONLY)")
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(fig_dir / f"{gse}_{prog}.png", dpi=145)
        plt.close(fig)
    return pd.DataFrame(outputs)


def run_dynamics(out: Path, session: requests.Session, offline: bool, gmt: Path) -> None:
    programs = read_gmt(gmt)
    qc_summary = []
    for gse in ["GSE243039", "GSE246591"]:
        meta_file = out / "metadata" / f"{gse}_samples.csv"
        if not meta_file.exists():
            print(f"[WARNING] {gse}: metadata missing. Run --mode metadata first.", file=sys.stderr)
            continue
        meta = pd.read_csv(meta_file).fillna({"donor_local":"", "group":"unknown"})
        control_label = "control" if gse == "GSE243039" else "reference_status_unverified"
        controls = meta.loc[(meta.group == control_label) & meta.donor_local.ne("") & meta.time_hours.notna()].copy()
        if len(controls) < 12 or controls.donor_local.nunique() < 3:
            print(f"[WARNING] {gse}: insufficient clearly mapped control/reference donor samples", file=sys.stderr)
            continue
        try:
            expr_path = locate_expression(out, session, gse, offline)
            all_expression = read_matrix(expr_path)
            symbolic, _ = gene_symbols_and_expression(all_expression, out, gse)
            proposed_cols = list(map(str, symbolic.columns))
            mapping = match_columns(proposed_cols, meta, gse, out)
            gsm_to_col = {gsm: col for col, gsm in mapping.items()}
            present_control = controls.loc[controls.gsm.isin(gsm_to_col)]
            if len(present_control) != len(controls):
                raise ValueError("Some control GSM columns are missing from expression matrix")
            # hard gate: subset before normalization, standardization or plots. NO case expression.
            expr = symbolic[[gsm_to_col[gsm] for gsm in present_control.gsm]].copy()
            expr.columns = list(present_control.gsm)
            scores, coverage = prepare_scores(expr, present_control, programs, EXPRESSION_KIND[gse])
            coverage.insert(0, "gse", gse)
            coverage.to_csv(out / "results" / f"{gse}_gene_coverage.csv", index=False)
            scored = scores.merge(present_control[["gsm", "group", "donor_local", "time_hours"]], on="gsm", how="left", validate="many_to_one")
            scored.to_csv(out / "results" / f"{gse}_reference_only_module_scores.csv", index=False)
            diag = plot_and_diagnose(scored, out, gse)
            diag.to_csv(out / "results" / f"{gse}_local_identifiability.csv", index=False)
            qc_summary.append({"gse":gse,"n_samples":len(present_control),"n_donors":present_control.donor_local.nunique(),
                               "n_programs_covered":scored.program.nunique(),"n_trajectory_diagnostics":len(diag),
                               "data_status":"CONTROL-ONLY" if gse=="GSE243039" else "REFERENCE-UNVERIFIED"})
            print(f"[Success] {gse}: {scored.program.nunique()} provisional programs; {len(diag)} diagnosable curves")
        except Exception as e:
            print(f"[WARNING] {gse} dynamics need manual check: {e}", file=sys.stderr)
            (out / "results" / f"{gse}_failure.txt").write_text(str(e)+"\n", encoding="utf-8")
    pd.DataFrame(qc_summary).to_csv(out / "results" / "dynamics_execution_summary.csv", index=False)
    (out / "results" / "INTERPRETATION.txt").write_text(
        "EXPLORATORY PILOT, NOT A SCIENTIFIC RESULT.\n"
        "Only control/reference profiles were analyzed. No patient expression was inspected.\n"
        "GSE246591 donor disease-free status remains unverified in GEO.\n"
        "Local corr(f,f') and condition number depend on curve smoothing, observation times, and nuisance intercept.\n"
        "High correlation or large condition number warn that time delay and amplitude are poorly distinguishable.\n"
        "Any rule such as |corr| > 0.95 is a HEURISTIC, not a test or established cutoff.\n"
        "Different protocols/platforms cannot be pooled into a single common clock without additional assumptions.\n"
        "Do not inspect case trajectories until final gene-program list, exclusion rules and model have been frozen.\n", encoding="utf-8")


def make_demo(out: Path, gmt: Path) -> None:
    """Offline deterministic synthetic smoke test; outputs cannot be used for biology."""
    rng = np.random.default_rng(42)
    programs = read_gmt(gmt)
    gse = "SYNTHETIC_DEMO"
    scores = []
    for program in list(programs)[:9]:
        for donor in range(4):
            offset = rng.normal(0, 0.1)
            for day in [0, 2, 4, 6, 8]:
                val = 1.1/(1+np.exp(-(day-3.7+offset))) + rng.normal(0, 0.055)
                scores.append({"program":program,"donor_local":f"demo{donor}","time_hours":day*24,"score":val, "gsm":f"SYN{donor}_{day}"})
    scores = pd.DataFrame(scores)
    (out / "results").mkdir(exist_ok=True, parents=True)
    diag = plot_and_diagnose(scores, out, gse)
    diag.to_csv(out / "results" / "SYNTHETIC_DEMO_identifiability.csv", index=False)
    (out / "results" / "DEMO_NOT_REAL_DATA.txt").write_text(
        "SYNTHETIC DEMONSTRATION ONLY. ZERO BIOLOGICAL CONCLUSIONS.\n", encoding="utf-8")
    print(f"[Demo] Wrote {len(diag)} synthetic identifiability diagnostics and figures")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["metadata", "dynamics", "all", "demo"], default="all")
    parser.add_argument("--out", type=Path, default=Path("paper7_results"))
    parser.add_argument("--offline", action="store_true", help="Read existing GEO files only (downloaded metadata header + matrices).")
    parser.add_argument("--gmt", type=Path, default=Path(__file__).parent / "config" / "programs_PROVISIONAL.gmt")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metadata").mkdir(exist_ok=True)
    (args.out / "results").mkdir(exist_ok=True)
    (args.out / "figures").mkdir(exist_ok=True)
    session = get_session()
    if args.mode in {"metadata", "all"}:
        metadata_audit(args.out, session, args.offline)
    if args.mode in {"dynamics", "all"}:
        run_dynamics(args.out, session, args.offline, args.gmt)
    if args.mode == "demo":
        make_demo(args.out, args.gmt)
    print(f"Outputs: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
