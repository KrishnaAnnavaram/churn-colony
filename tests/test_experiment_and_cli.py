import json

import pytest

from churn_colony.cli import main
from churn_colony.config import Settings, load_env_file
from churn_colony.cost import CostModel
from churn_colony.experiment import LeakageError, compare_selectors, evaluate, train
from churn_colony.schema import validate
from churn_colony.synthetic import generate


@pytest.fixture(scope="module")
def clean_mid():
    df, _ = validate(generate(1500, seed=21))
    return df


def test_train_picks_on_validation_and_scores_test_once(clean_mid):
    summary, bundle, parts = train(clean_mid, models=("logreg", "hgb"), seed=1, n_boot=100)
    assert set(summary["candidates"]) == {"logreg", "hgb"}
    best = summary["chosen_model"]
    assert summary["candidates"][best]["pr_auc"] == max(v["pr_auc"] for v in summary["candidates"].values())
    t = summary["test"]
    assert t["n"] == len(parts.test)
    lo, hi = t["roc_auc_ci95"]
    assert lo <= t["roc_auc"] <= hi
    assert 0.65 < t["roc_auc"] < 0.95
    assert t["profit_at_threshold"] >= t["reference_profits"]["contact_none"] - 1e-9 or summary["validation_profit"] >= 0
    assert 0 < bundle["threshold"] < 1


def test_train_stops_on_a_leak(clean_mid, monkeypatch):
    import churn_colony.experiment as exp

    monkeypatch.setattr(exp, "leakage_screen", lambda X, y, t: ["leaky_column"])
    with pytest.raises(LeakageError):
        train(clean_mid, models=("logreg",), n_boot=50)


def test_compare_selectors_runs_selection_inside_folds(clean_mid):
    table = compare_selectors(clean_mid, "logreg", ("none", "l1", "aco"), seed=0, folds=3,
                              aco_params={"n_ants": 4, "n_iterations": 2})
    assert list(table["selector"]) == ["none", "l1", "aco"]
    none_cols = table.loc[table.selector == "none", "columns_mean"].item()
    assert table.loc[table.selector == "l1", "columns_mean"].item() < none_cols
    assert (table["roc_auc_mean"] > 0.6).all()


def test_evaluate_on_new_rows(clean_mid):
    _, bundle, _ = train(clean_mid, models=("logreg",), n_boot=50)
    new, _ = validate(generate(600, seed=99))
    out = evaluate(bundle, new, n_boot=50)
    assert out["n"] == 600 and 0.6 < out["roc_auc"] < 0.95


def test_settings_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_COLONY_SEED", "7")
    monkeypatch.setenv("CHURN_COLONY_CONTACT_COST", "4.5")
    s = Settings.from_env()
    assert s.seed == 7 and s.contact_cost == 4.5 and s.data_path is None
    monkeypatch.setenv("CHURN_COLONY_OFFER_SUCCESS", "1.5")
    with pytest.raises(ValueError):
        Settings.from_env()
    env = tmp_path / ".env"
    env.write_text("CHURN_COLONY_VALUE_MONTHS=6\n# comment\nCHURN_COLONY_SEED=99\n", encoding="utf-8")
    monkeypatch.delenv("CHURN_COLONY_VALUE_MONTHS", raising=False)
    assert load_env_file(env) == 1  # SEED is already set and wins
    assert CostModel(value_months=6).customer_value([10]) == pytest.approx([60])


def test_cli_end_to_end(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("CHURN_COLONY_DATA", raising=False)
    csv = tmp_path / "syn.csv"
    assert main(["synth", "--rows", "900", "--seed", "2", "--out", str(csv)]) == 0
    assert main(["validate", str(csv)]) == 0
    run = tmp_path / "run"
    assert main(["train", "--data", str(csv), "--models", "logreg", "--out", str(run)]) == 0
    assert (run / "model.joblib").is_file() and (run / "model_card.md").is_file()
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["chosen_model"] == "logreg"
    capsys.readouterr()
    assert main(["evaluate", "--run", str(run), "--rows", "400", "--seed", "77"]) == 0
    assert json.loads(capsys.readouterr().out)["n"] == 400
    assert main(["explain", "--run", str(run), "--data", str(csv), "--customer", "SYN-00003"]) == 0
    assert "churn score" in capsys.readouterr().out
    assert main(["explain", "--run", str(run), "--data", str(csv), "--customer", "NOPE"]) == 1
    nolabel = tmp_path / "nolabel.csv"
    generate(50, seed=5).drop(columns=["Churn"]).to_csv(nolabel, index=False)
    out_csv = tmp_path / "scores.csv"
    assert main(["score", "--run", str(run), str(nolabel), "--out", str(out_csv)]) == 0
    assert out_csv.read_text(encoding="utf-8").startswith("customerID,churn_score,contact")
    assert main(["select", "--rows", "600", "--selectors", "none,l1", "--folds", "3"]) == 0
    assert main(["tune", "--rows", "500", "--model", "logreg", "--outer", "2", "--inner", "2", "--iter", "2"]) == 0
    assert main(["evaluate", "--run", str(tmp_path / "missing")]) == 1
