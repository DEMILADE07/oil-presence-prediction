# Prediction of Oil Presence Using Machine Learning

My solution for the **SPE DSEATS Africa 2026 Datathon** — predicting whether an unexplored
location holds oil (1) or not (0) from geological and seismic measurements.

The core idea is to treat this as a **petroleum-system** problem: oil only accumulates when a
good reservoir rock, a real trap, a nearby charge, and a supporting seismic indicator all line up
in the same place. I show from the data that discovery really does follow this "all at once" rule,
then build that logic into the features and the data cleaning.

## What's here

| File | Description |
|---|---|
| `TeamName_PythonCode.ipynb` | The full notebook: data audit, EDA, Part 1 and Part 2, predictions |
| `TeamName_Prediction_Part1.csv` | Test-set predictions from the standard workflow |
| `TeamName_Prediction_Part2.csv` | Test-set predictions from the physics-informed workflow |
| `PROJECT_WALKTHROUGH.md` | A plain-English, beginner-friendly guide to the whole project |
| `SOLUTION_OVERVIEW.md` | Approach summary, results, and submission checklist |
| `SLIDE_DECK_OUTLINE.md` | 10-slide presentation plan |
| `build_notebook.py` | Script that assembles the notebook |
| `requirements.txt` | Exact library versions |

> The competition datasets and the guidelines PDF are not included here, as they were provided by
> the organizers to registered participants.

## Approach in brief

- **Part 1 (standard ML):** reconcile the `Trap_Type` encoding, impute missing values, encode
  categories, and compare six models (Logistic Regression, Random Forest, Extra Trees,
  HistGradientBoosting, XGBoost, LightGBM) with 5-fold cross-validation, then average the best.
- **Part 2 (physics-informed):** correct five geological inconsistencies across `Trap_Type`,
  `Porosity` and `Permeability`, impute using rock physics, and add petroleum-system features
  (Reservoir Quality Index, Flow Zone Indicator, and a Chance-of-Success score).

## Results (5-fold cross-validation)

About **0.88 accuracy on the test-like rows** (those with a seismic score, like the test set) and
ROC-AUC around 0.81 — which sits at the realistic ceiling for this dataset, with full
interpretability (SHAP confirms the model learned reservoir, trap and seismic).

## How to run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook TeamName_PythonCode.ipynb   # then Run All
```

In Google Colab, upload the two competition CSVs first, then Runtime → Run all.
