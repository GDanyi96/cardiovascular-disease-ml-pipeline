# Cardiovascular Disease ML Pipeline

End-to-end binary classification pipeline on 70k patient records - gradient-boosting and Keras MLP baselines tuned via dual Bayesian optimisation strategies (hyperopt TPE, scikit-optimize random-forest surrogate).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GDanyi96/cardiovascular-disease-ml-pipeline/blob/main/cardiovascular_ml_pipeline.ipynb)

## What it does

A reproducible pipeline covering the full ML lifecycle: data validation, EDA, five classical baselines (XGBoost, LightGBM, AdaBoost, Logistic Regression, Ridge), a Keras MLP, Bayesian hyperparameter optimisation, and evaluation across ROC-AUC, F1, recall, and log-loss.

## Why it's interesting

The hyperparameter search uses **two complementary Bayesian optimisation strategies** on the same problem  `hyperopt`'s Tree-structured Parzen Estimator and `scikit-optimize`'s random-forest surrogate - letting you compare which surrogate model navigates the search space better for tabular medical data.

## How to run


```bash
git clone https://github.com/GDanyi96/cardiovascular-disease-ml-pipeline.git
cd cardiovascular-disease-ml-pipeline
pip install -r requirements.txt
jupyter notebook cardiovascular_ml_pipeline.ipynb
```


Or click the "Open in Colab" badge above, no local install needed.

## Data

[Cardiovascular Disease dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) — 70,000 records, 11 features, fetched automatically via `kagglehub`.

## Tech stack

`XGBoost` · `LightGBM` · `Keras` · `hyperopt` · `scikit-optimize` · `Plotly`
