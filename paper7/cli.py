"""Paper 7 command line: audit -> review -> freeze -> fit; or synthetic smoke tests."""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from .data import prepare_review, create_lock
from .simulation import run_simulation
from .workflow import run_locked_fit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    cmd = parser.add_subparsers(dest="command", required=True)
    sim = cmd.add_parser('simulate', help='Run explicitly SYNTHETIC mechanism experiments')
    sim.add_argument('--out', type=Path, default=Path('results/synthetic'))
    sim.add_argument('--reps', type=int, default=40)
    sim.add_argument('--seed', type=int, default=726)

    review_cmd = cmd.add_parser('prepare-review', help='Create UNVERIFIED manual review template from GEO metadata')
    review_cmd.add_argument('--results', type=Path, default=Path('geo_results'))
    review_cmd.add_argument('--out', type=Path, default=Path('geo_results/metadata/GSE243039_REVIEW.csv'))

    freeze = cmd.add_parser('freeze', help='Freeze confirmed donor labels and independently curated gene programs')
    freeze.add_argument('--results', type=Path, default=Path('geo_results'))
    freeze.add_argument('--review', type=Path, required=True)
    freeze.add_argument('--gmt', type=Path, required=True)
    freeze.add_argument('--out', type=Path, default=Path('geo_results/frozen_design.json'))
    freeze.add_argument('--curation-note', required=True, help='Biological literature rationale for locked programs')

    fit = cmd.add_parser('fit', help='Case/control dynamics after valid lock only')
    fit.add_argument('--results', type=Path, default=Path('geo_results'))
    fit.add_argument('--review', type=Path, required=True)
    fit.add_argument('--gmt', type=Path, required=True)
    fit.add_argument('--lock', type=Path, required=True)
    fit.add_argument('--out', type=Path, default=Path('results/real_stage1'))
    fit.add_argument('--online', action='store_true', help='Allow processed matrix download if absent')

    args = parser.parse_args()
    if args.command == 'simulate':
        report = run_simulation(args.out, reps=args.reps, seed=args.seed)
        print('SYNTHETIC ONLY (not GEO):\n' + report.to_string(index=False))
    elif args.command == 'prepare-review':
        path = args.results / 'metadata' / 'GSE243039_samples.csv'
        df = pd.read_csv(path, dtype={'gsm':str})
        template = prepare_review(df)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.exists():
            raise FileExistsError(f'Review template already exists: {args.out}')
        template.to_csv(args.out, index=False)
        print(f'UNVERIFIED manual-review template saved: {args.out}. No case data fitted.')
    elif args.command == 'freeze':
        lock = create_lock(args.review, args.gmt, args.results/'metadata'/'GSE243039_samples.csv',
                           args.out, args.curation_note)
        print(f'Frozen protocol saved: {args.out}. n_controls={lock["n_controls"]}, n_cases={lock["n_cases"]}')
    elif args.command == 'fit':
        fitted = run_locked_fit(args.results, args.gmt, args.review, args.lock,
                                args.out, offline=not args.online)
        print(f'Wrote {len(fitted)} donor/program rows in {args.out}; NOT clinical or causal conclusions')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
