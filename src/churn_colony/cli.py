"""Command line: synth | validate | train | evaluate | score | select | tune | explain."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from churn_colony import ID_COLUMN
from churn_colony.config import Settings, load_env_file
from churn_colony.cost import CostModel
from churn_colony.data import load
from churn_colony.experiment import compare_selectors, evaluate, load_run, save_run, score, train
from churn_colony.explain import explain_customer
from churn_colony.models import MODEL_NAMES
from churn_colony.report import model_card
from churn_colony.resample import RESAMPLING
from churn_colony.schema import FEATURE_COLUMNS, SchemaError, read_csv, validate
from churn_colony.selection import SELECTOR_NAMES
from churn_colony.synthetic import generate
from churn_colony.tuning import PARAM_SPACES, nested_cv


def _models(text: str) -> list[str]:
    names = [m.strip() for m in text.split(",") if m.strip()]
    bad = [m for m in names if m not in MODEL_NAMES]
    if bad or not names:
        raise argparse.ArgumentTypeError(f"unknown model(s) {bad}; choose from {MODEL_NAMES}")
    return names


def _data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", help="Telco CSV. Default: CHURN_COLONY_DATA, else synthetic rows")
    p.add_argument("--rows", type=int, default=5000, help="synthetic rows when no CSV is given")
    p.add_argument("--seed", type=int, help="random seed (default CHURN_COLONY_SEED or 42)")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="churn-colony", description=__doc__)
    ap.add_argument("--env-file", default=".env", help="optional KEY=VALUE file (default .env)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("synth", help="write a synthetic Telco-like CSV")
    p.add_argument("--rows", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data/synthetic_telco.csv")

    p = sub.add_parser("validate", help="check a CSV against the column contract")
    p.add_argument("csv")

    p = sub.add_parser("train", help="fit candidates, pick one on validation rows, score it once on test rows")
    _data_args(p)
    p.add_argument("--models", type=_models, default=["logreg", "hgb", "stack"], help="comma list")
    p.add_argument("--selector", choices=SELECTOR_NAMES, default="none")
    p.add_argument("--resampling", choices=RESAMPLING, default="none")
    p.add_argument("--out", help="run folder (default <CHURN_COLONY_OUTPUT>/latest)")

    p = sub.add_parser("evaluate", help="score a saved model on a labelled CSV")
    _data_args(p)
    p.add_argument("--run", required=True)

    p = sub.add_parser("score", help="write churn scores and contact flags for a CSV (no target needed)")
    p.add_argument("--run", required=True)
    p.add_argument("csv")
    p.add_argument("--out", default="-", help="output CSV, '-' for standard output")

    p = sub.add_parser("select", help="compare feature selectors with cross-validation")
    _data_args(p)
    p.add_argument("--model", choices=MODEL_NAMES, default="logreg")
    p.add_argument("--selectors", default=",".join(SELECTOR_NAMES))
    p.add_argument("--folds", type=int, default=5)

    p = sub.add_parser("tune", help="nested cross-validation of a randomised search")
    _data_args(p)
    p.add_argument("--model", choices=sorted(PARAM_SPACES), default="hgb")
    p.add_argument("--selector", choices=SELECTOR_NAMES, default="none")
    p.add_argument("--resampling", choices=RESAMPLING, default="none")
    p.add_argument("--outer", type=int, default=5)
    p.add_argument("--inner", type=int, default=3)
    p.add_argument("--iter", type=int, default=10)

    p = sub.add_parser("explain", help="per-customer effects for one customer ID")
    _data_args(p)
    p.add_argument("--run", required=True)
    p.add_argument("--customer", required=True, help="customerID")
    p.add_argument("--top", type=int, default=8)
    return ap


def _load(args, settings: Settings):
    seed = settings.seed if args.seed is None else args.seed
    path = args.data or settings.data_path
    df, report, source = load(path, seed=seed, synthetic_rows=args.rows)
    print(f"data: {source} | {report.summary()}", file=sys.stderr)
    return df, source, seed


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_env_file(args.env_file)
    try:
        settings = Settings.from_env()
        with threadpool_limits(limits=settings.threads):
            return _run(args, settings)
    except (SchemaError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _run(args, settings: Settings) -> int:
    cost = CostModel(settings.contact_cost, settings.offer_success_rate, settings.value_months)

    if args.cmd == "synth":
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        generate(args.rows, seed=args.seed).to_csv(out, index=False)
        print(f"wrote {args.rows} synthetic rows to {out}")
        return 0

    if args.cmd == "validate":
        _, report = validate(read_csv(args.csv))
        print(report.summary())
        return 0

    if args.cmd == "train":
        df, source, seed = _load(args, settings)
        summary, bundle, _ = train(df, args.models, args.selector, args.resampling, seed, cost)
        out = save_run(args.out or Path(settings.output_dir) / "latest", summary, bundle, model_card(summary, source))
        t = summary["test"]
        print(f"chosen model: {summary['chosen_model']} | threshold {summary['threshold']:.2f}")
        for name, v in summary["candidates"].items():
            print(f"  val {name:8s} roc_auc={v['roc_auc']:.3f} pr_auc={v['pr_auc']:.3f}")
        lo, hi = t["roc_auc_ci95"]
        print(f"test roc_auc={t['roc_auc']:.3f} [{lo:.3f}, {hi:.3f}] pr_auc={t['pr_auc']:.3f} "
              f"brier={t['brier']:.3f} lift@10%={t['lift_top10']:.2f} profit={t['profit_at_threshold']:.0f}")
        print(f"saved run to {out}")
        return 0

    if args.cmd == "evaluate":
        bundle = load_run(args.run)
        df, _, seed = _load(args, settings)
        print(json.dumps(evaluate(bundle, df, seed=seed), indent=2, default=float))
        return 0

    if args.cmd == "score":
        bundle = load_run(args.run)
        clean, _ = validate(read_csv(args.csv), require_target=False)
        out = score(bundle, clean)
        if args.out == "-":
            out.to_csv(sys.stdout, index=False)
        else:
            out.to_csv(args.out, index=False)
            print(f"wrote {len(out)} scores to {args.out}")
        return 0

    if args.cmd == "select":
        df, _, seed = _load(args, settings)
        names = [s.strip() for s in args.selectors.split(",") if s.strip()]
        table = compare_selectors(df, args.model, names, seed, args.folds)
        print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return 0

    if args.cmd == "tune":
        df, _, seed = _load(args, settings)
        from churn_colony.schema import split_xy

        X, y = split_xy(df)
        res = nested_cv(X, y, args.model, args.selector, args.resampling, seed, args.outer, args.inner, args.iter)
        print(json.dumps(res, indent=2, default=float))
        return 0

    if args.cmd == "explain":
        bundle = load_run(args.run)
        df, _, _ = _load(args, settings)
        row = df[df[ID_COLUMN] == args.customer]
        if row.empty:
            raise KeyError(f"customer {args.customer!r} is not in the data")
        table = explain_customer(bundle["pipeline"], row.loc[:, list(FEATURE_COLUMNS)], bundle["reference"])
        print(f"customer {args.customer}: churn score {table.attrs['score']:.3f} "
              f"(threshold {bundle['threshold']:.2f})")
        with pd.option_context("display.width", 120):
            print(table.head(args.top).to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
        return 0
    return 2  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
