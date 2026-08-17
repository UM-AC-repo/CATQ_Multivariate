# CATQ Multivariate

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

Analysis code and reproducible research materials for a Maastricht University study of sex differences and trait-specific correlates of autistic camouflaging.

## Overview

This repository contains the code used to examine variation in the three domains of the Camouflaging Autistic Traits Questionnaire (CAT-Q): **Compensation**, **Masking**, and **Assimilation**. The analyses evaluate whether CAT-Q scores differ between women and men, whether any differences persist after accounting for empathizing, systemizing, and autistic traits, and whether these trait associations vary by sex.

The work accompanies the manuscript:

> Monteiro S, Ambraß L, de Sousa Fernandes Perna E, Stauder J. *Rethinking the Boundaries of Camouflaging: A Multivariate Analysis of Sex, Empathizing, Systemizing, and Autistic Traits.* Manuscript in preparation, 2026.

## Study and analytical sample

The study was conducted at Maastricht University using cross-sectional online-survey data. The processed dataset contained 195 participants. The primary women–men models included 193 participants: 145 women and 48 men. Two participants who selected another sex/gender response category were retained in the completed datasets but were not included in the binary women–men models because that group was too small for reliable estimation.

Missing questionnaire values were handled using multiple imputation. Twenty completed datasets were generated; observed values were retained unchanged, and only missing values were replaced with plausible draws. Models were fitted separately in every completed dataset, and coefficients and covariance matrices were pooled using Rubin's rules.

## Analytical workflow

The notebook implements three sequential model stages:

1. **Model 1:** CAT-Q outcomes as a function of sex and age.
2. **Model 2:** Model 1 with Empathy Quotient (EQ), Systemizing Quotient (SQ), and 10-item Autism Spectrum Quotient (AQ-10) added.
3. **Model 3:** Model 2 with sex × EQ, sex × SQ, and sex × AQ-10 interactions added.

Compensation, Masking, and Assimilation are modelled jointly using multivariate multiple linear regression. CAT-Q total is analysed separately because it is derived from the three subscales. The workflow includes:

- HC3 robust covariance estimation;
- pooling across 20 imputed datasets using Rubin's rules;
- multivariate Wald tests;
- Benjamini–Hochberg false-discovery-rate correction;
- sex-specific simple slopes and adjusted predictions;
- model-fit and Shapley variance decomposition;
- nonlinearity, overlap, collinearity, cutoff, and influence diagnostics;
- leave-one-participant-out stability analysis; and
- generation of manuscript figures and tables.

## Repository structure

```text
CATQ_Multivariate/
├── code/       Analysis notebooks and supporting scripts
├── data/       Data documentation and authorized analysis inputs
├── output/     Generated figures, tables, and diagnostic summaries
├── CITATION.cff
├── LICENSE
└── README.md
```

Generated files should normally be recreated from the analysis code rather than edited manually.

## Running the analysis

The analysis was developed in Python and can be run in Google Colab or a compatible Jupyter environment.

Principal dependencies include:

```text
numpy
pandas
scipy
matplotlib
seaborn
scikit-learn
python-docx
openpyxl
```

Place the authorized imputation package in `data/`, open the principal notebook in `code/`, and execute the workflow in order. The notebook is stateful: supplementary, figure, and table cells may consume fitted objects created by earlier modelling cells and should not be assumed to function independently. Before creating a release, restart the kernel and verify the complete workflow in a fresh session.

## Data protection

This repository must not contain identifiable or pseudonymized participant-level information unless public sharing is explicitly permitted by the participant consent, ethics approval, institutional policy, and applicable data-protection law. The original Qualtrics/SPSS export, contact information, and direct or indirect identifiers should not be committed.

The `data/` directory should contain only data that have been approved for distribution, synthetic or example data, or documentation describing how qualified researchers may request access. The repository license does **not** override consent, ethical, contractual, privacy, or data-protection restrictions.

## Citation

If you use the code, analytical workflow, figures, or other licensed materials, please cite:

> Monteiro S, Ambraß L, de Sousa Fernandes Perna E, Stauder J. *Rethinking the Boundaries of Camouflaging: A Multivariate Analysis of Sex, Empathizing, Systemizing, and Autistic Traits.* 2026.

Machine-readable citation metadata are provided in [`CITATION.cff`](CITATION.cff). Update the preferred citation with the journal, volume, pages, and DOI after publication.

## License

Except where otherwise stated, the original code, documentation, and non-sensitive outputs in this repository are licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Participant-level data, personal data, third-party materials, and any content for which the repository contributors do not control the necessary rights are excluded from this license unless explicitly identified otherwise. See [`LICENSE`](LICENSE).

## Contact

Sara Monteiro  
Department of Cognitive Neuroscience, Faculty of Psychology and Neuroscience, Maastricht University  
[s.monteiro@alumni.maastrichtuniversity.nl](mailto:s.monteiro@alumni.maastrichtuniversity.nl)
