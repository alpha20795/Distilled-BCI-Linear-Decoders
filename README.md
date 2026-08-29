# Neural Decoding Experiments

Four Jupyter notebooks exploring feature selection, data pooling, knowledge distillation, and cross-session generalization in BCI neural decoding.

## Experiments

- **exp1/** - Feature selection method comparison (XGBoost, DNC, Lasso)
- **exp2/** - Training data pooling strategies impact
- **exp3/** - Knowledge distillation with Mamba encoder and quantization
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
