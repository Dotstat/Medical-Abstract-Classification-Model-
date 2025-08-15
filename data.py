import re, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _ensure_nltk():
    for pkg in ['stopwords','wordnet']:
        try:
            nltk.data.find(f'corpora/{pkg}')
        except LookupError:
            nltk.download(pkg, quiet=True)

_ensure_nltk()
STOP = set(stopwords.words('english'))
LEMM = WordNetLemmatizer()

def clean_text(text: str) -> str:
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return ' '.join(LEMM.lemmatize(w) for w in text.split() if w not in STOP)

def load_df(url: str):
    df = pd.read_csv(url)
    df = (df.drop_duplicates(subset='medical_abstract')
            .dropna(subset=['medical_abstract','condition_label']))
    df['adjusted_label'] = df['condition_label'].astype(int) - 1
    df = df[df['adjusted_label'].between(0,4)]
    df['cleaned_abstract'] = df['medical_abstract'].apply(clean_text)
    df = df[df['cleaned_abstract'].str.split().apply(len) >= 10]
    return df
