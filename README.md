# Neural Decoding Experiments

Four Jupyter notebooks exploring feature selection, data pooling, knowledge distillation, and cross-session generalization in BCI neural decoding.

## Experiments

- **exp1/** - Joint feature & classifier selection (XGBoost/DNC/Lasso features × LR/LDA classifiers)
- **exp2/** - Pooling strategy comparison (Separate day-specific models vs. Chunk pooling vs. Total cross-day pooling) with inter-session MMD distribution analysis
- **exp3/** - Mamba encoder trained across sessions for session-invariant learning. OMP identifies sparse features. Student: Lasso (L1) feature selection + L2-penalized logistic regression classifier. Includes 32/16/8/4-bit quantization
- **zero_shot/** - Zero-shot to few-shot generalization across recording sessions



## Requirements

- NumPy, SciPy, scikit-learn, matplotlib
- PyTorch (Exp 3 & zero_shot)
- XGBoost (Exp 1)
- Mamba (optional for Exp 3 & zero_shot; falls back to GRU)


## Dataset

The dataset used in this project was obtained from the Dryad Digital Repository:

> Willett, Francis; Avansino, Donald; Hochberg, Leigh et al. (2021). Data from: High-performance brain-to-text communication via handwriting [Dataset]. Dryad. https://doi.org/10.5061/dryad.wh70rxwmv

**Source:** [Dryad Dataset](https://doi.org/10.5061/dryad.wh70rxwmv)
