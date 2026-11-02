# Paper 7 — Transcriptomic Timing and Identifiability

Research code for auditing temporal-response identifiability
and inferring experimentally documented exposure duration
from RNA-seq data.

**Exposure duration is not biological transcription onset.**

## Reproduce the development analyses

Requires Python 3.11 or 3.12 and internet access.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r reproducibility/requirements.txt
python reproducibility/replay.py
```

The runner executes the original V112–V120 scripts, downloads
public GEO/eLife data, saves logs and checks the headline
numerical results.

See [Reproducibility](reproducibility/README.md).

## OnsetAudit prototype

Open `onsetaudit/index.html` locally in a browser.

The first version inspects metadata only.
It does not predict biological onset.

## Evidence boundaries

GSE194200: experimentally recorded drug-exposure durations.

GSE149428: second study, examined during method development.

V120: two-stage model selected after viewing V119 outcomes.
Development results: 79/90 active correct-and-accepted,
29/30 DMSO controls rejected.

GSE214965: raw count matrix verified; locked external
validation still pending.

No independent validation of true transcription onset,
95% coverage or methodological novelty is established.

Original versioned scripts are preserved at the repository
root to retain code hashes and historical dependencies.

This repository is for research, not clinical decisions.
Licensing and final manuscript citation are pending.
