import inspect
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from .vectorizers import tfidf_base, make_count

def specialist_feats_for_k_r2(keep_words_k, err_words_k, kw_pair_k, k,
                              use_char, char_ngrams, char_min_df,
                              kw_weight, err_weight, pair_weight, char_weight):
    transformers = [
        ('tfidf', tfidf_base()),
        ('kw_seed', make_count(keep_words_k[k]))
    ]
    weights = {'tfidf': 1.0, 'kw_seed': kw_weight}

    if err_words_k.get(k):
        transformers.append(('kw_err', make_count(err_words_k[k])))
        weights['kw_err'] = err_weight

    if kw_pair_k and kw_pair_k.get(k):
        transformers.append(('kw_pair', make_count(kw_pair_k[k])))
        weights['kw_pair'] = pair_weight

    if use_char:
        transformers.append(('char', TfidfVectorizer(
            analyzer='char_wb', ngram_range=char_ngrams, min_df=char_min_df, sublinear_tf=True)))
        weights['char'] = char_weight

    return FeatureUnion(transformer_list=transformers, transformer_weights=weights)

def make_specialist_r2(keep_words_k, err_words_k, kw_pair_k, k, use_calibration, **cfg):
    feats = specialist_feats_for_k_r2(keep_words_k, err_words_k, kw_pair_k, k,
                                      cfg['use_char'], cfg['char_ngrams'], cfg['char_min_df'],
                                      cfg['kw_weight'], cfg['err_weight'], cfg['pair_weight'], cfg['char_weight'])
    base_lr = LogisticRegression(max_iter=3000, class_weight='balanced',
                                 C=1.0, solver='lbfgs', random_state=42)
    if use_calibration:
        if "estimator" in inspect.signature(CalibratedClassifierCV).parameters:
            clf = CalibratedClassifierCV(estimator=base_lr, method='sigmoid', cv=3)
        else:
            clf = CalibratedClassifierCV(base_estimator=base_lr, method='sigmoid', cv=3)
    else:
        clf = base_lr
    return Pipeline([('feats', feats), ('clf', clf)])
