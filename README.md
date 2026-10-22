# Fair clustering for categorical data: intersectional fairness via correspondence analysis

## Problem
Fair-Multicluster (Santos-Mangudo & Heras, 2023) adds a per-attribute total-variation
(TV) fairness penalty to categorical clustering. With several protected attributes,
that penalty only sees each attribute's *marginal* distribution. A disparity that
exists only in a *combination* (e.g. Female × Minority × Rural) can leave marginal TV
at zero while clusters are badly skewed on the joint table.

## Method
We add a second penalty on the joint protected-category × cluster contingency table:

    L = L_sim + λ_tv · F_marginal + λ_ca · Φ²,    Φ² = χ²/n on the joint table

- λ_ca = 0 reproduces the TV-fair baseline exactly (verified in code).
- Φ² equals the sum of squared canonical correlations between cluster membership and
  the joint protected category (classical CCA ↔ correspondence-analysis identity),
  so it needs no per-attribute weights. Derivation in `docs/MATH.md`.
- Optimiser: steepest-descent local search over multicluster moves. Order-independent;
  each move updates only two contingency-table columns.

## Results (synthetic leakage study)
Fixed λ_tv = 1, sweep λ_ca, 5 replicate datasets per leakage order, n = 8000:

| Leakage order | Φ² reduction vs TV-fair |
|---|---|
| 1 (marginal)  | 15.2% |
| 2 (pairwise)  | 16.1% |
| 3 (3-way)     | 22.8% |

Similarity (cluster quality) stays ≈ 0.73–0.74 throughout. Plots in `results/`.

## Limitations
- Local search; no global-optimality guarantee.
- λ behaves in steps because the similarity score is coarse; always sweep λ.
- Joint table grows multiplicatively with protected attributes, so it gets sparse.
- Evidence so far is simulation only.
- The Multicluster baseline is a simple greedy proxy

## Run it
```r
install.packages(c("FactoMineR","factoextra","dplyr","tidyr","ggplot2","scales"))
source("R/engine.R"); source("R/datagen.R"); source("R/plots.R")
rmarkdown::render("analysis/01_demo_simulation.Rmd")
```

## References
- Santos-Mangudo & Heras (2023), *A fair-multicluster approach to clustering of categorical data*, CEJOR 31(2).
- Greenacre (1984), *Theory and Applications of Correspondence Analysis*.
- Kearns et al. (2018), *Preventing Fairness Gerrymandering*.
