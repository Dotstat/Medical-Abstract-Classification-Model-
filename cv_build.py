import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .vectorizers import tfidf_base
from .gating import split_kw_sets, contains_kw_for_k, split_kw_pair_sets, contains_kw_any_for_k
from .specialists import make_specialist

def make_splits(y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros_like(y), y))

def build_cv_blobs(X, y, keep_words_k, err_words_k, classes, splits=None, use_calibration=True,
                   use_char=True, char_ngrams=(3,5), char_min_df=3,
                   kw_weight=2.2, err_weight=2.4, char_weight=0.8):
    if splits is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = skf.split(X, y)

    KW_UNI_k, KW_BI_k = split_kw_sets(keep_words_k, classes)
    fold_blobs = []

    for tr, te in splits:
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        base_pipe = Pipeline([
            ('tfidf', tfidf_base()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=3000, class_weight='balanced',
                C=1.0, multi_class='multinomial', random_state=42))
        ])
        base_pipe.fit(X_tr, y_tr)
        base_proba = base_pipe.predict_proba(X_te)
        base_pred  = base_proba.argmax(1)

        p_spec = {}; kw_mask = {}
        for k in classes:
            y_tr_bin = (y_tr == k).astype(int)
            spec_pipe = make_specialist(keep_words_k, err_words_k, k, use_calibration,
                                        use_char=use_char, char_ngrams=char_ngrams, char_min_df=char_min_df,
                                        kw_weight=kw_weight, err_weight=err_weight, char_weight=char_weight)
            spec_pipe.fit(X_tr, y_tr_bin)
            p_spec[k] = spec_pipe.predict_proba(X_te)[:, 1]
            kw_mask[k] = np.array([contains_kw_for_k(t, k, KW_UNI_k, KW_BI_k) for t in X_te.values], dtype=bool)

        fold_blobs.append({
            "base_proba": base_proba,
            "base_pred":  base_pred,
            "y_te":       y_te,
            "texts":      X_te.values,
            "p_spec":     p_spec,
            "kw_mask":    kw_mask
        })
    return fold_blobs

def build_cv_blobs_r2(X, y, keep_words_k, err_words_k, kw_pair_k, classes, splits=None, **cfg):
    if splits is None:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = skf.split(X, y)

    from .specialists import make_specialist as _make_spec
    from .specialists import LogisticRegression, Pipeline, tfidf_base
    from .gating import split_kw_sets, split_kw_pair_sets, contains_kw_any_for_k

    KW_UNI_k, KW_BI_k = split_kw_sets(keep_words_k, classes)
    P_UNI_k,  P_BI_k  = split_kw_pair_sets(kw_pair_k, classes)
    fold_blobs = []

    for tr, te in splits:
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]

        base_pipe = Pipeline([
            ('tfidf', tfidf_base()),
            ('clf', LogisticRegression(
                solver='lbfgs', max_iter=3000, class_weight='balanced',
                C=1.0, multi_class='multinomial', random_state=42))
        ])
        base_pipe.fit(X_tr, y_tr)
        base_proba = base_pipe.predict_proba(X_te)
        base_pred  = base_proba.argmax(1)

        # reuse make_specialist but with kw_pair by temporarily merging into err_words_k-like weight via cfg
        from .specialists_r2 import make_specialist_r2
        p_spec = {}; kw_mask = {}
        for k in classes:
            y_tr_bin = (y_tr == k).astype(int)
            spec_pipe = make_specialist_r2(keep_words_k, err_words_k, kw_pair_k, k,
                                           cfg.get('use_calibration', True),
                                           use_char=cfg.get('use_char', True),
                                           char_ngrams=cfg.get('char_ngrams', (3,5)),
                                           char_min_df=cfg.get('char_min_df', 3),
                                           kw_weight=cfg.get('kw_weight', 2.2),
                                           err_weight=cfg.get('err_weight', 2.4),
                                           pair_weight=cfg.get('pair_weight', 3.0),
                                           char_weight=cfg.get('char_weight', 0.8))
            spec_pipe.fit(X_tr, y_tr_bin)
            p_spec[k] = spec_pipe.predict_proba(X_te)[:, 1]
            kw_mask[k] = np.array(
                [contains_kw_any_for_k(t, k, KW_UNI_k, KW_BI_k, P_UNI_k, P_BI_k) for t in X_te.values],
                dtype=bool
            )

        fold_blobs.append({
            "base_proba": base_proba,
            "base_pred":  base_pred,
            "y_te":       y_te,
            "texts":      X_te.values,
            "p_spec":     p_spec,
            "kw_mask":    kw_mask
        })
    return fold_blobs
