"""Model card in Markdown, written from the run summary."""

from __future__ import annotations


def _f(x, nd=3) -> str:
    return f"{x:.{nd}f}"


def model_card(summary: dict, source: str) -> str:
    t = summary["test"]
    th = t["at_threshold"]
    lines = [
        "# Model card: churn-colony",
        "",
        f"- Data source: `{source}`",
        f"- Splits (stratified, seed {summary['seed']}): {summary['splits']}",
        f"- Train churn rate: {_f(summary['train_churn_rate'])}",
        f"- Selector: `{summary['selector']}`, resampling: `{summary['resampling']}`",
        f"- Chosen model: `{summary['chosen_model']}` ({summary['chosen_by']})",
        f"- Model columns after selection: {len(summary['model_columns'])}",
        "",
        "## Candidates (validation rows)",
        "",
        "| Model | ROC-AUC | PR-AUC | Brier | Fit seconds |",
        "|---|---|---|---|---|",
    ]
    for name, v in summary["candidates"].items():
        lines.append(f"| {name} | {_f(v['roc_auc'])} | {_f(v['pr_auc'])} | {_f(v['brier'])} | {v['fit_seconds']} |")
    lo, hi = t["roc_auc_ci95"]
    plo, phi = t["pr_auc_ci95"]
    lines += [
        "",
        "## Test rows (used once)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| ROC-AUC | {_f(t['roc_auc'])} (95 % CI {_f(lo)}-{_f(hi)}) |",
        f"| PR-AUC | {_f(t['pr_auc'])} (95 % CI {_f(plo)}-{_f(phi)}) |",
        f"| Base rate | {_f(t['base_rate'])} |",
        f"| Brier score | {_f(t['brier'])} |",
        f"| ECE (10 bins) | {_f(t['ece'])} |",
        f"| Lift in top 10 % | {_f(t['lift_top10'], 2)} |",
        f"| Threshold (from validation profit) | {_f(summary['threshold'], 2)} |",
        f"| Precision / recall at threshold | {_f(th['precision'])} / {_f(th['recall'])} |",
        f"| Share of customers contacted | {_f(th['contacted_share'])} |",
        f"| Profit at threshold | {t['profit_at_threshold']:.0f} |",
        f"| Profit if all are contacted | {t['reference_profits']['contact_all']:.0f} |",
        "",
        "## Top raw features (permutation importance, ROC-AUC drop on test rows)",
        "",
        "| Feature | Importance |",
        "|---|---|",
    ]
    for row in summary["importance"]:
        lines.append(f"| {row['feature']} | {_f(row['importance'], 4)} |")
    c = summary["cost_model"]
    lines += [
        "",
        "## Cost model",
        "",
        f"contact cost {c['contact_cost']}, offer success rate {c['success_rate']}, value months {c['value_months']}.",
        "",
        "## Limits and responsible use",
        "",
        "- The scores rank customers for a retention offer. They are not a decision about a person.",
        "- A person must review the contact list. Do not use protected traits to target or exclude customers.",
        "- The model learns from past customers. If prices, products or the customer mix change, train again.",
        "- Synthetic-data results show that the code works. They do not predict results on real customers.",
    ]
    return "\n".join(lines) + "\n"
