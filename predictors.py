import os, json, joblib, numpy as np
from .data import clean_text
from .gating import split_kw_sets, contains_kw_for_k, split_kw_pair_sets, contains_kw_any_for_k

class FusedPredictor:
    def __init__(self, model_dir: str):
        with open(os.path.join(model_dir, "meta.json")) as f:
            meta = json.load(f)
        self.classes   = [int(k) for k in meta["classes"]]
        self.T_high    = {int(k): v for k, v in meta["T_high"].items()}
        self.Delta     = {int(k): v for k, v in meta["Delta"].items()}
        self.RequireKW = {int(k): v for k, v in meta["RequireKW"].items()}
        self.baseline  = joblib.load(os.path.join(model_dir, "baseline.pkl"))
        self.specs     = {k: joblib.load(os.path.join(model_dir, f"specialist_{k}.pkl"))
                          for k in self.classes}
        with open(os.path.join(model_dir, "keep_words.json")) as f:
            self.keep_words_k = {int(k): set(v) for k, v in json.load(f).items()}
        self.KW_UNI_k, self.KW_BI_k = split_kw_sets(self.keep_words_k, self.classes)

    def _kw_present(self, text: str, k: int) -> bool:
        return contains_kw_for_k(text, k, self.KW_UNI_k, self.KW_BI_k)

    def predict_fused(self, texts):
        texts_clean = [clean_text(t) for t in texts]
        base_proba = self.baseline.predict_proba(texts_clean)
        base_pred  = base_proba.argmax(1)
        p_spec = {k: self.specs[k].predict_proba(texts_clean)[:, 1] for k in self.classes}
        kw_mask = {k: np.array([self._kw_present(t, k) for t in texts_clean], dtype=bool) for k in self.classes}

        fused = base_pred.copy()
        for i in range(len(texts_clean)):
            b = base_pred[i]
            cands = []
            for k in self.classes:
                if k == b: continue
                cond = (p_spec[k][i] >= self.T_high[k]) and ((base_proba[i].max() - base_proba[i, k]) <= self.Delta[k])
                if self.RequireKW[k]:
                    cond = cond and kw_mask[k][i]
                if cond:
                    cands.append((k, p_spec[k][i]))
            if cands:
                fused[i] = max(cands, key=lambda t: t[1])[0]
        return fused

def load_model(model_dir: str) -> FusedPredictor:
    return FusedPredictor(model_dir)

class FusedPredictorR2(FusedPredictor):
    def __init__(self, model_dir: str):
        super().__init__(model_dir)
        pair_path = os.path.join(model_dir, "kw_pair.json")
        if os.path.exists(pair_path):
            with open(pair_path) as f:
                self.kw_pair_k = {int(k): set(v) for k, v in json.load(f).items()}
        else:
            self.kw_pair_k = {k:set() for k in self.classes}
        self.KW_ANY_UNI_k, self.KW_ANY_BI_k = {}, {}
        for k in self.classes:
            seeds = self.keep_words_k.get(k, set())
            pairs = self.kw_pair_k.get(k, set())
            uni = {w for w in seeds if ' ' not in w} | {w for w in pairs if ' ' not in w}
            bi  = {w for w in seeds if ' ' in  w} | {w for w in pairs if ' ' in  w}
            self.KW_ANY_UNI_k[k] = uni
            self.KW_ANY_BI_k[k]  = bi

    def _kw_present(self, text: str, k: int) -> bool:
        toks = text.split()
        if any(w in self.KW_ANY_UNI_k[k] for w in toks): return True
        for i in range(len(toks)-1):
            if f"{toks[i]} {toks[i+1]}" in self.KW_ANY_BI_k[k]: return True
        return False

class TwoStagePredictor:
    def __init__(self, dir_r0, dir_r1):
        self.r0 = load_model(dir_r0)
        self.r1 = load_model(dir_r1)

    def predict_top2(self, texts):
        p0 = self.r0.predict_fused(texts)
        p1 = self.r1.predict_fused(texts)
        out = []
        for a, b in zip(p1, p0):
            out.append((int(a), int(b) if b != a else None))
        import numpy as np
        return np.array(out, dtype=object)

class ThreeStagePredictor:
    def __init__(self, dir_r0, dir_r1, dir_r2):
        self.r0 = load_model(dir_r0)
        self.r1 = load_model(dir_r1)
        self.r2 = FusedPredictorR2(dir_r2)

    def predict_top2(self, texts):
        p2 = self.r2.predict_fused(texts)
        p1 = self.r1.predict_fused(texts)
        p0 = self.r0.predict_fused(texts)
        out = []
        for a, b, c in zip(p2, p1, p0):
            top = [int(a)]
            if b != a: top.append(int(b))
            elif c != a: top.append(int(c))
            else: top.append(None)
            out.append(tuple(top))
        import numpy as np
        return np.array(out, dtype=object)
