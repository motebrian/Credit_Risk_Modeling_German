# Credit Risk Modeling — German Credit Data

A machine learning project that predicts whether a loan applicant is a **Good** or **Bad** credit risk, using the [German Credit Data with Risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk) dataset. Includes full exploratory data analysis, model comparison, and an interactive Streamlit app for live predictions.

## Dataset

1,000 loan applicants with the following fields:

| Column | Description |
|---|---|
| Age | Applicant age (years) |
| Sex | male / female |
| Job | 0 = unskilled & non-resident, 1 = unskilled & resident, 2 = skilled, 3 = highly skilled |
| Housing | own / rent / free |
| Saving accounts | little / moderate / quite rich / rich |
| Checking account | little / moderate / rich |
| Credit amount | Loan amount requested |
| Duration | Loan term (months) |
| Purpose | Reason for the loan (car, education, business, etc.) |
| Risk | Target: good / bad |

`Saving accounts` and `Checking account` have missing values (18.3% and 39.4% respectively); rows with missing data are dropped during cleaning, leaving 522 applicants for modeling.

## Project workflow

All analysis lives in [`Credit_Risk_Modeling_German.ipynb`](Credit_Risk_Modeling_German.ipynb):

1. **Data cleaning** — download via `kagglehub`, drop rows with missing `Saving accounts` / `Checking account`.
2. **Exploratory analysis** — distributions, boxplots (outlier check + by-Risk comparison), a correlation heatmap of the numeric features, a violin plot of credit amount by savings level, and categorical breakdowns by `Risk`.
3. **Feature encoding** — `Sex`, `Housing`, `Saving accounts`, and `Checking account` are label-encoded and each encoder is persisted (`*_encoder.pkl`) for reuse at inference time. `Purpose` is excluded from the model features.
4. **Modeling** — a stratified 80/20 train/test split, then `GridSearchCV` tuning of four classifiers with class-imbalance handling (`class_weight="balanced"` / `scale_pos_weight`).
5. **Model export** — the best-performing model is saved as `extra_trees_credit_model.pkl`.

### Model comparison (test accuracy)

| Model | Accuracy |
|---|---|
| Decision Tree | 0.581 |
| Random Forest | 0.619 |
| **Extra Trees** | **0.648** |
| XGBoost | 0.648 |

Extra Trees was selected as the final model (tied with XGBoost, but simpler to tune). Accuracy is modest, which is expected given the dataset shrinks to 522 rows after dropping missing values and the two accounts fields carry a lot of the predictive signal.

## Streamlit app

[`app.py`](app.py) provides a form for entering an applicant's details and returns a live Good/Bad risk prediction from the trained Extra Trees model.

### Run it locally

```bash
pip install streamlit pandas scikit-learn joblib
streamlit run app.py
```

Then open the local URL Streamlit prints (default: `http://localhost:8501`).

## Repository structure

```
Credit_Risk_Modeling_German.ipynb   # EDA + model training notebook
app.py                              # Streamlit prediction app
german_credit_data.csv              # Raw dataset
extra_trees_credit_model.pkl        # Trained model
Sex_encoder.pkl                     # Label encoders used by app.py
Housing_encoder.pkl
Saving accounts_encoder.pkl
Checking account_encoder.pkl
target_encoder.pkl                  # Risk label encoder
```

## Dependencies

- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- xgboost
- joblib
- kagglehub (dataset download only)
- streamlit (app only)
