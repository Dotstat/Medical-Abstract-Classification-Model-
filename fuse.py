import numpy as np
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                             precision_recall_fscore_support)

def fuse_with_params(blob, classes, T_high, Delta, RequireKW):
    base_proba, base_pred = blob["base_proba"], blob["base_pred"]
    p_spec, kw_mask = blob["p_spec"], blob["kw_mask"]
    y_te, texts = blob["y_te"], blob["texts"]
    fused = base_pred.copy()

    for i in range(len(texts)):
        b = base_pred[i]
        cands = []
        for k in classes:
            if k == b: continue
            cond = (p_spec[k][i] >= T_high[k]) and ((base_proba[i].max() - base_proba[i, k]) <= Delta[k])
            if RequireKW[k]:
                cond = cond and kw_mask[k][i]
            if cond:
                cands.append((k, p_spec[k][i]))
        if cands:
            fused[i] = max(cands, key=lambda t: t[1])[0]
    return y_te, fused

def evaluate(fold_blobs, classes, T_high, Delta, RequireKW):
    y_all, p_all = [], []
    for blob in fold_blobs:
        y_te, pred = fuse_with_params(blob, classes, T_high, Delta, RequireKW)
        y_all.extend(y_te); p_all.extend(pred)
    y_all = np.array(y_all); p_all = np.array(p_all)
    pr, rc, f1, _ = precision_recall_fscore_support(y_all, p_all, labels=classes)
    return {
        "acc": accuracy_score(y_all, p_all),
        "macroF1": f1_score(y_all, p_all, average='macro'),
        "pr": pr, "rc": rc, "f1": f1,
        "y": y_all, "pred": p_all
    }

def tune_thresholds(fold_blobs, classes, objective, T_CAND, DELTA_CAND, KW_REQ_CAND, N_ITERS_CA):
    T_high = {k: 0.42 for k in classes}
    Delta  = {k: 0.05 for k in classes}
    ReqKW  = {k: True  for k in classes}

    best = evaluate(fold_blobs, classes, T_high, Delta, ReqKW)
    print(f"\n[Init] Acc={best['acc']:.4f} MacroF1={best['macroF1']:.4f}")
    def score(m): return m['macroF1'] if objective=='macroF1' else m['acc']

    for it in range(N_ITERS_CA):
        improved = False
        print(f"\n-- Coordinate ascent pass {it+1}/{N_ITERS_CA} --")
        for k in classes:
            cur = (score(best), dict(T_high), dict(Delta), dict(ReqKW), best)
            for t in T_CAND:
                for d in DELTA_CAND:
                    for rq in KW_REQ_CAND:
                        T_try, D_try, R_try = dict(T_high), dict(Delta), dict(ReqKW)
                        T_try[k], D_try[k], R_try[k] = t, d, rq
                        cand = evaluate(fold_blobs, classes, T_try, D_try, R_try)
                        if score(cand) > cur[0] + 1e-6:
                            cur = (score(cand), T_try, D_try, R_try, cand)
            if cur[0] > score(best) + 1e-6:
                T_high, Delta, ReqKW, best = cur[1], cur[2], cur[3], cur[4]
                improved = True
                print(f"  * class {k} → Acc={best['acc']:.4f} MacroF1={best['macroF1']:.4f} "
                      f"(T={T_high[k]}, D={Delta[k]}, KW={ReqKW[k]})")
        if not improved:
            print("  (no further improvement)"); break

    print("\n=== GUARDED FUSION (per-class, tuned) ===")
    print(classification_report(best['y'], best['pred'], digits=4))
    print(f"Accuracy: {best['acc']:.4f}")
    print(f"Macro F1: {best['macroF1']:.4f}")
    return T_high, Delta, ReqKW, best

def union_accuracy(fold_blobs0, T0, D0, R0, fold_blobs1, T1, D1, R1, classes):
    from sklearn.metrics import accuracy_score
    import numpy as np
    y_all, p0_all, p1_all = [], [], []
    for b0, b1 in zip(fold_blobs0, fold_blobs1):
        y0, pred0 = fuse_with_params(b0, classes, T0, D0, R0)
        y1, pred1 = fuse_with_params(b1, classes, T1, D1, R1)
        assert np.array_equal(y0, y1)
        y_all.append(y0); p0_all.append(pred0); p1_all.append(pred1)
    y = np.concatenate(y_all); p0 = np.concatenate(p0_all); p1 = np.concatenate(p1_all)
    acc0 = accuracy_score(y, p0); acc1 = accuracy_score(y, p1)
    acc_or = np.mean((y == p0) | (y == p1))
    only_r1_fix = np.mean((y != p0) & (y == p1))
    both_wrong  = np.mean((y != p0) & (y != p1))
    print(f"\n[Two-pass union] R0 acc={acc0:.4f} | R1 acc={acc1:.4f} | Union acc={acc_or:.4f} (lift {acc_or-acc0:+.4f})")
    print(f"  R1 fixes (R0 wrong, R1 right): {only_r1_fix*100:.1f}% | Both wrong: {both_wrong*100:.1f}%")
    return acc_or

def union_accuracy_three(fold_blobs0, T0, D0, R0,
                         fold_blobs1, T1, D1, R1,
                         fold_blobs2, T2, D2, R2,
                         classes):
    from sklearn.metrics import accuracy_score
    import numpy as np
    y_all, p0_all, p1_all, p2_all = [], [], [], []
    for b0, b1, b2 in zip(fold_blobs0, fold_blobs1, fold_blobs2):
        y0, pred0 = fuse_with_params(b0, classes, T0, D0, R0)
        y1, pred1 = fuse_with_params(b1, classes, T1, D1, R1)
        y2, pred2 = fuse_with_params(b2, classes, T2, D2, R2)
        assert np.array_equal(y0, y1) and np.array_equal(y0, y2)
        y_all.append(y0); p0_all.append(pred0); p1_all.append(pred1); p2_all.append(pred2)

    y  = np.concatenate(y_all)
    p0 = np.concatenate(p0_all); p1 = np.concatenate(p1_all); p2 = np.concatenate(p2_all)

    acc0 = accuracy_score(y, p0)
    acc1 = accuracy_score(y, p1)
    acc2 = accuracy_score(y, p2)
    acc_union2 = np.mean((y == p0) | (y == p1))
    acc_union3 = np.mean((y == p0) | (y == p1) | (y == p2))

    only_r2_fix_over_r0r1 = np.mean((y != p0) & (y != p1) & (y == p2))
    still_all_wrong       = np.mean((y != p0) & (y != p1) & (y != p2))

    print(f"\n[Three-pass union] R0={acc0:.4f} | R1={acc1:.4f} | R2={acc2:.4f} | "
          f"Union-2={acc_union2:.4f} → Union-3={acc_union3:.4f} (Δ {acc_union3-acc_union2:+.4f})")
    print(f"  R2 unique fixes (R0&R1 wrong, R2 right): {only_r2_fix_over_r0r1*100:.1f}% | "
          f"All three wrong: {still_all_wrong*100:.1f}%")
    return acc_union3
