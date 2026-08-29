# Neural Decoding Experiments

Four Jupyter notebooks exploring feature selection, data pooling, knowledge distillation, and cross-session generalization in BCI neural decoding.

## Experiments

- **exp1/** - Joint feature & classifier selection (XGBoost/DNC/Lasso features × LR/LDA classifiers)
- **exp2/** - Pooling strategy comparison (Separate day-specific models vs. Chunk pooling vs. Total cross-day pooling) with inter-session MMD distribution analysis
- **exp3/** - Mamba encoder trained across sessions for session-invariant learning. OMP identifies sparse features. Student: Lasso (L1) feature selection + L2-penalized logistic regression classifier. Includes 32/16/8/4-bit quantization
- **0shot/** - Zero-shot to few-shot generalization across recording sessions

## Running

```bash
cd exp1 && jupyter notebook experiment-1-with-prefilter.ipynb
cd exp2 && jupyter notebook experiment-2.ipynb
cd exp3 && jupyter notebook experiment-3.ipynb
cd 0shot && jupyter notebook few-shot.ipynb
```

## Requirements

- NumPy, SciPy, scikit-learn, matplotlib
- PyTorch (Exp 3 & 4)
- XGBoost (Exp 1)
- Mamba (optional for Exp 3 & 4; falls back to GRU)

## Data

Neural recordings from T5 participant (192 channels, 10 recording days). Update `data_folder` paths in notebooks to point to your local dataset.
