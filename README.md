# Rigid 3D Point Registration (ICP + RANSAC)

This repository contains:
- a Python implementation,
- a C++20 implementation (Armadillo),
- a technical accuracy report,
- and an algorithm write-up (LaTeX + PDF).

---

## Repository structure

```
registration_repo/
├── src/
│   ├── registration_modern_icp_ransac.py
│   └── registration_modern_icp_ransac_commented_armadillo.cpp
├── report/
│   └── technical_report_accuracy.md
├── docs/
│   ├── registration_algorithm_explained.tex
│   └── registration_algorithm_explained.pdf
└── README.md
```

---

## Algorithm summary

The pipeline uses two stages:

1. RANSAC-based pose initialization (optional, when correspondence candidates are available)
2. Trimmed point-to-point ICP for pose refinement

The full equations and step-by-step flow are in:
- `docs/registration_algorithm_explained.pdf`

---

## Build and run (C++)

Requires Armadillo.

```bash
g++ -O3 -std=c++20 src/registration_modern_icp_ransac_commented_armadillo.cpp -larmadillo -o reg
./reg 3 21
```

Arguments:
- first: number of trials
- second: base random seed

---

## Accuracy report

- `report/technical_report_accuracy.md`

The report summarizes accuracy numbers and the synthetic settings used to obtain them for both Python and C++.
