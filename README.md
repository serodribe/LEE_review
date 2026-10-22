# LEE_review

Classical studies of the **look-elsewhere effect (LEE)** in bump-hunt searches.

The notebook [`notebooks/experiments.ipynb`](notebooks/experiments.ipynb) covers:

1. **Classical bump hunt**
   - Null distribution of the test statistic $q$ and local $p$-values, fixed vs. scanned mass
   - Gross & Vitells correction (upcrossings)
   - Gross–Vitells vs. Bonferroni trial factors
   - Dependence of the trial factor on the signal template width
2. **Multi-channel search**
   - Particle decay channels
   - Unknown physics process
   - Bonferroni, Šidák and Benjamini–Hochberg corrections

Figures are written to `output/plots/classic/`.

## Usage

```bash
pip install -r requirements.txt
cd notebooks
jupyter notebook experiments.ipynb
```

Run the notebook from inside `notebooks/` so the relative output path resolves correctly.
