import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from .vectorizers import tfidf_base

def build_keep_words(X, y, classes, excl_margin, keep_top):
    seed_pipe = Pipeline([
        ('tfidf', tfidf_base()),
        ('clf', LogisticRegression(
            solver='lbfgs', max_iter=3000, class_weight='balanced',
            C=1.0, multi_class='multinomial', random_state=42))
    ])
    seed_pipe.fit(X, y)
    feats = np.array(seed_pipe.named_steps['tfidf'].get_feature_names_out())
    clf  = seed_pipe.named_steps['clf']
    coefs = clf.coef_
    cls_idx = {c: i for i, c in enumerate(clf.classes_)}

    keep_words_k = {}
    for k in classes:
        row = cls_idx[k]
        ck = coefs[row]; order = np.argsort(ck)[::-1]
        kws = []
        for j in order:
            m_other = np.max(np.delete(coefs[:, j], row))
            if (ck[j] - m_other) >= excl_margin:
                kws.append(feats[j])
            if len(kws) >= keep_top:
                break
        keep_words_k[k] = set(kws)
        print(f"[Seeds] class {k}: {len(keep_words_k[k])} keep_words")
    return keep_words_k
