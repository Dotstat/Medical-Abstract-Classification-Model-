from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def tfidf_base():
    return TfidfVectorizer(
        tokenizer=str.split, preprocessor=None, token_pattern=None,
        lowercase=False, strip_accents='unicode',
        sublinear_tf=True, ngram_range=(1,2),
        min_df=2, max_df=0.95
    )

def make_count(vocab):
    return CountVectorizer(
        vocabulary=sorted(vocab),
        analyzer='word', ngram_range=(1,2), binary=True,
        tokenizer=str.split, preprocessor=None, token_pattern=None,
        lowercase=False
    )
