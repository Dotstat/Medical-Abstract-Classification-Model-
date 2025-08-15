import argparse, numpy as np
from . import config as C
from .data import load_df
from .seeds import build_keep_words
from .cv_build import build_cv_blobs, build_cv_blobs_r2
from .fuse import tune_thresholds, union_accuracy, union_accuracy_three, fuse_with_params
from .mining import (collect_oof_preds, mine_error_terms_per_class,
                     collect_union_masks, confusion_pairs_from_both_wrong, mine_pair_terms)
from .export import train_and_export, train_and_export_r2

def cmd_train():
    df = load_df(C.DATA_URL)
    X = df['cleaned_abstract']; y = df['adjusted_label'].values
    CLASSES = sorted(np.unique(y).tolist())

    keep_words_k = build_keep_words(X, y, CLASSES, C.EXCL_MARGIN, C.KEEP_TOP)
    err_words_k_0 = {k:set() for k in CLASSES}

    fb0 = build_cv_blobs(X, y, keep_words_k, err_words_k_0, CLASSES,
                         use_calibration=C.USE_CALIBRATION, use_char=C.USE_CHAR,
                         char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF,
                         kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT, char_weight=C.CHAR_WEIGHT)
    T0, D0, R0, best0 = tune_thresholds(fb0, CLASSES, C.OBJECTIVE, C.T_CAND, C.DELTA_CAND, C.KW_REQ_CAND, C.N_ITERS_CA)
    print(f"[Round 0] Acc={best0['acc']:.4f} MacroF1={best0['macroF1']:.4f}")

    oof0 = collect_oof_preds(fb0, CLASSES, T0, D0, R0, fuse_with_params)
    err_words_k_1 = mine_error_terms_per_class(oof0, CLASSES, topn=C.ERR_TOP, min_df=C.ERR_MIN_DF, pref_bigrams=C.PREF_BIGRAMS)

    fb1 = build_cv_blobs(X, y, keep_words_k, err_words_k_1, CLASSES,
                         use_calibration=C.USE_CALIBRATION, use_char=C.USE_CHAR,
                         char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF,
                         kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT, char_weight=C.CHAR_WEIGHT)
    T1, D1, R1, best1 = tune_thresholds(fb1, CLASSES, C.OBJECTIVE, C.T_CAND, C.DELTA_CAND, C.KW_REQ_CAND, C.N_ITERS_CA)
    print(f"[Round 1] Acc={best1['acc']:.4f} MacroF1={best1['macroF1']:.4f}")

    _ = union_accuracy(fb0, T0, D0, R0, fb1, T1, D1, R1, CLASSES)

    df_union = collect_union_masks(fb0, T0, D0, R0, fb1, T1, D1, R1, CLASSES, fuse_with_params)
    top_pairs, df_bw = confusion_pairs_from_both_wrong(df_union, topn=C.PAIR_TOPN)
    print("\n[Both-wrong] Top confusion pairs (y -> r1):", top_pairs)
    kw_pair_k = {k:set() for k in CLASSES}
    for pair in top_pairs:
        terms = mine_pair_terms(df_union, df_bw, pair, topn=C.PAIR_TERMS_PER_CLASS, min_df=C.PAIR_MIN_DF, pref_bigrams=C.PREF_BIGRAMS)
        kw_pair_k[pair[0]].update(terms)
    print("[kw_pair sizes]:", {k: len(v) for k,v in kw_pair_k.items()})

    from . import config as Cfg
    fb2 = build_cv_blobs_r2(X, y, keep_words_k, err_words_k_1, kw_pair_k, CLASSES,
                            use_calibration=C.USE_CALIBRATION, use_char=C.USE_CHAR,
                            char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF,
                            kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT,
                            pair_weight=C.KW_PAIR_WEIGHT, char_weight=C.CHAR_WEIGHT)
    T2, D2, R2, best2 = tune_thresholds(fb2, CLASSES, C.OBJECTIVE, C.T_CAND, C.DELTA_CAND, C.KW_REQ_CAND, C.N_ITERS_CA)
    print(f"[Round 2] Acc={best2['acc']:.4f} MacroF1={best2['macroF1']:.4f}")

    _ = union_accuracy_three(fb0, T0, D0, R0, fb1, T1, D1, R1, fb2, T2, D2, R2, CLASSES)

    train_and_export(X, y, keep_words_k, err_words_k_0, CLASSES, T0, D0, R0, out_dir=C.R0_DIR,
                     use_calibration=C.USE_CALIBRATION, kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT,
                     use_char=C.USE_CHAR, char_weight=C.CHAR_WEIGHT, char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF)
    train_and_export(X, y, keep_words_k, err_words_k_1, CLASSES, T1, D1, R1, out_dir=C.R1_DIR,
                     use_calibration=C.USE_CALIBRATION, kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT,
                     use_char=C.USE_CHAR, char_weight=C.CHAR_WEIGHT, char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF)
    train_and_export_r2(X, y, keep_words_k, err_words_k_1, kw_pair_k, CLASSES, T2, D2, R2, out_dir=C.R2_DIR,
                        use_calibration=C.USE_CALIBRATION, kw_weight=C.KW_WEIGHT, err_weight=C.ERR_WEIGHT,
                        pair_weight=C.KW_PAIR_WEIGHT, use_char=C.USE_CHAR, char_weight=C.CHAR_WEIGHT,
                        char_ngrams=C.CHAR_NGRAMS, char_min_df=C.CHAR_MIN_DF)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cmd", choices=["train"])
    args = parser.parse_args()
    if args.cmd == "train":
        cmd_train()

if __name__ == "__main__":
    main()
