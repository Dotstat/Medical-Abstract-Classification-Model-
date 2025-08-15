import os, json, joblib
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .vectorizers import tfidf_base
from .specialists import make_specialist
from .specialists_r2 import make_specialist_r2

def train_and_export(X, y, keep_words_k, err_words_k, classes, T_high, Delta, RequireKW, out_dir,
                     use_calibration=True, **cfg):
    os.makedirs(out_dir, exist_ok=True)

    baseline = Pipeline([
        ('tfidf', tfidf_base()),
        ('clf', LogisticRegression(
            solver='lbfgs', max_iter=3000, class_weight='balanced',
            C=1.0, multi_class='multinomial', random_state=42))
    ])
    baseline.fit(X, y)
    joblib.dump(baseline, os.path.join(out_dir, "baseline.pkl"))

    for k in classes:
        y_bin = (y == k).astype(int)
        spec = make_specialist(keep_words_k, err_words_k, k, use_calibration, **cfg)
        spec.fit(X, y_bin)
        joblib.dump(spec, os.path.join(out_dir, f"specialist_{k}.pkl"))

    with open(os.path.join(out_dir, "keep_words.json"), "w") as f:
        json.dump({str(k): sorted(list(keep_words_k[k])) for k in classes}, f, indent=2)
    with open(os.path.join(out_dir, "err_words.json"), "w") as f:
        json.dump({str(k): sorted(list(err_words_k.get(k, set()))) for k in classes}, f, indent=2)

    meta = {"classes": classes, "T_high": T_high, "Delta": Delta, "RequireKW": RequireKW}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Exported] → '{out_dir}'")

def train_and_export_r2(X, y, keep_words_k, err_words_k, kw_pair_k, classes, T_high, Delta, RequireKW, out_dir,
                        use_calibration=True, **cfg):
    os.makedirs(out_dir, exist_ok=True)

    baseline = Pipeline([
        ('tfidf', tfidf_base()),
        ('clf', LogisticRegression(
            solver='lbfgs', max_iter=3000, class_weight='balanced',
            C=1.0, multi_class='multinomial', random_state=42))
    ])
    baseline.fit(X, y)
    joblib.dump(baseline, os.path.join(out_dir, "baseline.pkl"))

    for k in classes:
        y_bin = (y == k).astype(int)
        spec = make_specialist_r2(keep_words_k, err_words_k, kw_pair_k, k, use_calibration, **cfg)
        spec.fit(X, y_bin)
        joblib.dump(spec, os.path.join(out_dir, f"specialist_{k}.pkl"))

    import json
    with open(os.path.join(out_dir, "keep_words.json"), "w") as f:
        json.dump({str(k): sorted(list(keep_words_k[k])) for k in classes}, f, indent=2)
    with open(os.path.join(out_dir, "err_words.json"), "w") as f:
        json.dump({str(k): sorted(list(err_words_k.get(k, set()))) for k in classes}, f, indent=2)
    with open(os.path.join(out_dir, "kw_pair.json"), "w") as f:
        json.dump({str(k): sorted(list(kw_pair_k.get(k, set()))) for k in classes}, f, indent=2)

    meta = {"classes": classes, "T_high": T_high, "Delta": Delta, "RequireKW": RequireKW}
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[Exported] → '{out_dir}'")
