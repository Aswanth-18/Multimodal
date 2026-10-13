"""Train + evaluate CAD / LAD / LCX / RCA classifiers.
Run:  python -m backend.ml.train
Outputs: backend/models/*.joblib, backend/models/schema.json, results/metrics.{json,csv,md}, results/*.png
"""
import json, os, warnings
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.base import clone
from xgboost import XGBClassifier
from .data import load
from .features import F, TARGETS

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
MODELS, RES = ROOT / "backend" / "models", ROOT / "results"
SEED = 42

def candidates():
    return {
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000, random_state=SEED)),
        "RandomForest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt", random_state=SEED, n_jobs=-1),
        "XGBoost": XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.04, subsample=0.8, colsample_bytree=0.7,
                                 reg_lambda=3, eval_metric="logloss", random_state=SEED, n_jobs=4),
    }

def metrics(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return dict(accuracy=accuracy_score(y, yh), precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh), f1=f1_score(y, yh), roc_auc=roc_auc_score(y, p))


def write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, suffix=".tmp", delete=False, encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=float, ensure_ascii=False)
        tmp = fh.name
    os.replace(tmp, path)


def write_joblib(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, suffix=".tmp", delete=False) as fh:
        tmp = fh.name
    try:
        joblib.dump(payload, tmp)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def main():
    MODELS.mkdir(exist_ok=True); RES.mkdir(exist_ok=True)
    X, Y = load()
    print("Features:", X.shape[1], "| rows:", len(X))
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
    rows, best = [], {}
    for t in TARGETS:
        y = Y[t].values
        for name, est in candidates().items():
            fold = []
            for tr, te in cv.split(X, y):
                m = clone(est).fit(X.iloc[tr], y[tr])
                fold.append(metrics(y[te], m.predict_proba(X.iloc[te])[:, 1]))
            df = pd.DataFrame(fold)
            row = dict(target=t.upper(), model=name)
            for k in df: row[k] = df[k].mean(); row[k + "_std"] = df[k].std()
            rows.append(row)
            print(f"{t.upper():4s} {name:19s} AUC {row['roc_auc']:.3f}±{row['roc_auc_std']:.3f}  acc {row['accuracy']:.3f}  F1 {row['f1']:.3f}")
        sub = [r for r in rows if r["target"] == t.upper()]
        best[t] = max(sub, key=lambda r: r["roc_auc"])["model"]
    M = pd.DataFrame(rows)
    M.to_csv(RES / "metrics.csv", index=False)

    # final fit: calibrated best model per target + base model for SHAP; out-of-fold curves for plots
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    oof_all, summary = {}, {}
    for t in TARGETS:
        y, name = Y[t].values, best[t]
        cal = CalibratedClassifierCV(clone(candidates()[name]), method="sigmoid", cv=5)
        oof = cross_val_predict(cal, X, y, cv=skf, method="predict_proba")[:, 1]  # calibrated, out-of-fold
        oof_all[t] = oof
        summary[t] = dict(model=name, **metrics(y, oof), n_pos=int(y.sum()), n=len(y))
        cal.fit(X, y)
        base = clone(candidates()[name]).fit(X, y)
        payload = dict(calibrated=cal, base=base, kind=name, features=list(X.columns),
                      background=X.sample(100, random_state=SEED))
        write_joblib(MODELS / f"{t}.joblib", payload)
    write_json(RES / "metrics.json", dict(cv_all=rows, selected=summary))

    # schema for the API / UI
    sch = []
    for c in X.columns:
        f = dict(F[c]); f.update(name=c, default=float(X[c].median()), min=float(X[c].min()), max=float(X[c].max()))
        if f["options"]: f["options"] = {str(k): v for k, v in f["options"].items()}
        sch.append(f)
    write_json(MODELS / "schema.json", sch)

    # example patients (real, correctly-classified out-of-fold rows)
    P = pd.DataFrame(oof_all)
    low = P[(Y.sum(axis=1) == 0)].sort_values("cad").index[0]
    both = Y[(Y.cad == 1) & (Y.lad == 1) & (Y.lcx == 1) & (Y.rca == 1)].index
    high = P.loc[both].mean(axis=1).sort_values().index[-1]
    lad_only = Y[(Y.lad == 1) & (Y.lcx == 0) & (Y.rca == 0)].index
    mid = P.loc[lad_only].mean(axis=1).sort_values().index[-3]
    presets = {k: {c: float(X.loc[i, c]) for c in X.columns} for k, i in
               [("Low-risk example", low), ("LAD-only example", mid), ("High-risk example", high)]}
    write_json(MODELS / "presets.json", presets)

    # report tables + plots
    with open(RES / "metrics.md", "w", encoding="utf-8") as fh:
        fh.write("## Cross-validated performance (5-fold x 3 repeats, mean ± std)\n\n| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            star = " ★" if best[r["target"].lower()] == r["model"] else ""
            fh.write(f"| {r['target']} | {r['model']}{star} | " + " | ".join(f"{r[k]:.3f} ± {r[k+'_std']:.3f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]) + " |\n")
        fh.write("\n★ = selected (highest mean ROC-AUC). Selected models are then probability-calibrated (sigmoid).\n\n## Calibrated out-of-fold performance of selected models (5-fold)\n\n| Target | Model | Accuracy | Precision | Recall | F1 | ROC-AUC |\n|---|---|---|---|---|---|---|\n")
        for t, s in summary.items():
            fh.write(f"| {t.upper()} | {s['model']} | " + " | ".join(f"{s[k]:.3f}" for k in ["accuracy", "precision", "recall", "f1", "roc_auc"]) + " |\n")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for t in TARGETS:
        fpr, tpr, _ = roc_curve(Y[t], oof_all[t]); ax.plot(fpr, tpr, label=f"{t.upper()} (AUC {summary[t]['roc_auc']:.2f})")
    ax.plot([0, 1], [0, 1], "k--", lw=.8); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Out-of-fold ROC curves"); ax.legend(); fig.tight_layout(); fig.savefig(RES / "roc_curves.png", dpi=150)
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.4))
    for a, t in zip(axs, TARGETS):
        cm = confusion_matrix(Y[t], (oof_all[t] >= .5).astype(int)); a.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm): a.text(j, i, v, ha="center", va="center")
        a.set_title(t.upper()); a.set_xticks([0, 1]); a.set_yticks([0, 1]); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
    fig.tight_layout(); fig.savefig(RES / "confusion_matrices.png", dpi=150)
    print("\nSelected:", {t: (s['model'], round(s['roc_auc'], 3)) for t, s in summary.items()})

if __name__ == "__main__":
    main()
