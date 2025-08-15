# Disease Abstract Classification — Guarded Fusion (R0 → R1 → R2)

Three-round pipeline for medical abstract classification:
- **R0**: seed specialists from baseline LR coefficients
- **R1**: error-mined keywords per class
- **R2**: both-wrong pair mining + seed∪pair gating

## Quickstart
```bash
pip install -r requirements.txt
python scripts/train_r0_r1_r2.py
```
Artifacts are saved to `models/` (gitignored).
