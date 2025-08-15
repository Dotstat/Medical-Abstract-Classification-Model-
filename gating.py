def split_kw_sets(keep_words_k, classes):
    KW_UNI_k = {k: {w for w in keep_words_k[k] if ' ' not in w} for k in classes}
    KW_BI_k  = {k: {w for w in keep_words_k[k] if ' ' in  w} for k in classes}
    return KW_UNI_k, KW_BI_k

def contains_kw_for_k(text, k, KW_UNI_k, KW_BI_k):
    toks = text.split()
    if any(w in KW_UNI_k[k] for w in toks): return True
    for i in range(len(toks)-1):
        if f"{toks[i]} {toks[i+1]}" in KW_BI_k[k]: return True
    return False

def split_kw_pair_sets(kw_pair_k, classes):
    P_UNI_k = {k: {w for w in kw_pair_k.get(k, set()) if ' ' not in w} for k in classes}
    P_BI_k  = {k: {w for w in kw_pair_k.get(k, set()) if ' ' in  w} for k in classes}
    return P_UNI_k, P_BI_k

def contains_kw_any_for_k(text, k, KW_UNI_k, KW_BI_k, P_UNI_k, P_BI_k):
    toks = text.split()
    if any((w in KW_UNI_k[k]) or (w in P_UNI_k[k]) for w in toks):
        return True
    for i in range(len(toks) - 1):
        bg = f"{toks[i]} {toks[i+1]}"
        if (bg in KW_BI_k[k]) or (bg in P_BI_k[k]):
            return True
    return False
