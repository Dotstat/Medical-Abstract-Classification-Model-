# ==== Core config ====
N_SPLITS        = 5
EXCL_MARGIN     = 0.12
KEEP_TOP        = 250

KW_WEIGHT       = 2.2
USE_CHAR        = True
CHAR_WEIGHT     = 0.8
CHAR_NGRAMS     = (3, 5)
CHAR_MIN_DF     = 3
USE_CALIBRATION = True

# Threshold tuner grid
T_CAND      = [0.30, 0.33, 0.36, 0.39, 0.42, 0.45]
DELTA_CAND  = [0.03, 0.05, 0.07, 0.09]
KW_REQ_CAND = [True, False]
N_ITERS_CA  = 3
OBJECTIVE   = 'macroF1'     # or 'acc'

# Error-mining (for R1)
ERR_TOP      = 120
ERR_MIN_DF   = 4
ERR_WEIGHT   = 2.4
PREF_BIGRAMS = True

# R2
PAIR_TOPN = 6
PAIR_TERMS_PER_CLASS = 180
PAIR_MIN_DF = 4
KW_PAIR_WEIGHT = 3.0

# Dirs
R0_DIR = "models/fusion_model_r0"
R1_DIR = "models/fusion_model_r1"
R2_DIR = "models/fusion_model_r2"

# Data
DATA_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vTF5Nksdyyd3xGRjz7L5kgxkTe9FaM1gSII2J6MfbA9S0RoENMneq6FpjmVFSQmE8pNAc1N0dTcXerT/pub?output=csv"
