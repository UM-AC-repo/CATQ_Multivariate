# CATQ Multivariate Study

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626.svg)](https://jupyter.org/)

Analysis code and reproducible research materials for a Maastricht University study of sex differences and trait-specific correlates of autistic camouflaging.

## Overview

This repository contains the code used to examine variation in the three domains of the Camouflaging Autistic Traits Questionnaire (CAT-Q): **Compensation**, **Masking**, and **Assimilation**. The analyses evaluate whether CAT-Q scores differ between women and men, whether any differences persist after accounting for empathizing, systemizing, and autistic traits, and whether these trait associations vary by sex.

The work accompanies the manuscript:

> Monteiro et al. (2026). *Rethinking the Boundaries of Camouflaging: A Multivariate Analysis of Sex, Empathizing, Systemizing, and Autistic Traits.* 

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

## Repository contents

| Directory | Contents |
|---|---|
| [`code/`](code/) | Contains the complete analysis notebook, including modelling, sensitivity analyses, figures, and tables. It also contains the standalone impute script used to generate the multiply imputed datasets. |
| [`data/`](data/) | Contains the processed source dataset of 195 participants and the ZIP package with the 20 completed imputed datasets used by the analysis notebook. |


## License

Except where otherwise stated, the original code, documentation, and non-sensitive outputs in this repository are licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

Participant-level data, personal data, third-party materials, and any content for which the repository contributors do not control the necessary rights are excluded from this license unless explicitly identified otherwise. See [`LICENSE`](LICENSE).

<p align="center">
  <img src="assets/um-logo.jpg" alt="Maastricht University logo" width="300">
</p>
## Contact

Sara Monteiro  
Department of Cognitive Neuroscience, Faculty of Psychology and Neuroscience, Maastricht University  
[s.monteiro@alumni.maastrichtuniversity.nl](mailto:s.monteiro@alumni.maastrichtuniversity.nl)
