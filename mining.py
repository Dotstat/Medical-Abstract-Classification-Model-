import numpy as np, pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

def collect_oof_preds(fold_blobs, classes, T_high, Delta, RequireKW, fuse_fn):
    rows=[]
    for b in fold_blobs:
        y_te, fused = fuse_fn(b, classes, T_high, Delta, RequireKW)
        base = b["base_pred"]; texts = b["texts"]
        for i in range(len(texts)):
            rows.append({"y": int(y_te[i]), "pred": int(fused[i]),
                         "base": int(base[i]), "text": texts[i]})
    return pd.DataFrame(rows)

def mine_error_terms_per_class(df_oof, classes, topn=120, min_df=4, pref_bigrams=True):
    err_words_k = {k:set() for k in classes}
    for k in classes:
        fn = df_oof[(df_oof.y==k) & (df_oof.pred!=k)]["text"]
        tp = df_oof[(df_oof.y==k) & (df_oof.pred==k)]["text"]
        if len(fn) < 30 or len(tp) < 30:
            continue
        vec = CountVectorizer(
            ngram_range=(1,2), min_df=min_df,
            tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False
        )
        corpus = pd.concat([fn, tp], ignore_index=True)
        X = vec.fit_transform(corpus)
        n_fn = len(fn)
        fw = (X[:n_fn].sum(0).A1 + 1)
        ft = (X[n_fn:].sum(0).A1 + 1)
        score = np.log(fw / ft)
        terms = vec.get_feature_names_out()
        order = np.argsort(score)[::-1]
        picked, bigrams = [], 0
        for j in order:
            t = terms[j]
            if pref_bigrams and ' ' not in t and bigrams < topn//2:
                continue
            picked.append(t)
            if ' ' in t: bigrams += 1
            if len(picked) >= topn:
                break
        err_words_k[k].update(picked)
    return err_words_k

def collect_union_masks(fold_blobs0, T0, D0, R0, fold_blobs1, T1, D1, R1, classes, fuse_fn):
    rows = []
    for b0, b1 in zip(fold_blobs0, fold_blobs1):
        y0, p0 = fuse_fn(b0, classes, T0, D0, R0)
        y1, p1 = fuse_fn(b1, classes, T1, D1, R1)
        assert np.array_equal(y0, y1)
        texts = b0["texts"]
        for i in range(len(texts)):
            rows.append({
                "y": int(y0[i]),
                "r0": int(p0[i]),
                "r1": int(p1[i]),
                "both_wrong": bool((y0[i] != p0[i]) and (y0[i] != p1[i])),
                "text": texts[i]
            })
    import pandas as pd
    return pd.DataFrame(rows)

def confusion_pairs_from_both_wrong(df_union, topn=6):
    sub = df_union[df_union["both_wrong"]].copy()
    if sub.empty:
        return [], sub
    pairs = list(zip(sub["y"], sub["r1"]))
    vc = pd.Series(pairs).value_counts()
    top_pairs = [tuple(p) for p in vc.head(topn).index.tolist()]
    return top_pairs, sub

def mine_pair_terms(df_union_all, df_both_wrong, pair, topn=180, min_df=4, pref_bigrams=True):
    y_true, y_pred = pair
    miss = df_both_wrong[(df_both_wrong["y"]==y_true) & (df_both_wrong["r1"]==y_pred)]["text"]
    hit  = df_both_wrong[(df_both_wrong["y"]==y_true) & (df_both_wrong["r1"]==y_true)]["text"]
    if len(hit) < 20:
        alt = df_union_all[(df_union_all["y"]==y_true) & (df_union_all["r1"]==y_true)]["text"]
        hit = alt if len(alt) >= 20 else hit

    if len(miss) < 20 or len(hit) < 20:
        return []

    vec = CountVectorizer(
        ngram_range=(1,2), min_df=max(min_df, int(0.01*len(miss))),
        tokenizer=str.split, preprocessor=None, token_pattern=None, lowercase=False
    )
    import pandas as pd, numpy as np
    corpus = pd.concat([miss, hit], ignore_index=True)
    X = vec.fit_transform(corpus)
    n_miss = len(miss)

    fm = (X[:n_miss].sum(0).A1 + 1)
    fh = (X[n_miss:].sum(0).A1 + 1)
    score = np.log(fm / fh)

    terms = vec.get_feature_names_out()
    order = np.argsort(score)[::-1]
    picked, bigrams = [], 0
    for j in order:
        t = terms[j]
        if pref_bigrams and ' ' not in t and bigrams < topn//2:
            continue
        picked.append(t)
        if ' ' in t: bigrams += 1
        if len(picked) >= topn: break

    if len(picked) < topn:
        for j in order:
            t = terms[j]
            if t in picked: continue
            picked.append(t)
            if len(picked) >= topn: break
    return picked
