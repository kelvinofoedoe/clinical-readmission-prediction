# Diabetic Readmission Prediction

A replication and extension of Strack et al. (2014), *"Impact of HbA1c Measurement on Hospital Readmission Rates: Analysis of 70,000 Clinical Database Patient 
Records"*, using the same Diabetes 130-US Hospitals dataset.

The original paper used logistic regression purely for statistical inference and reports no predictive accuracy. This project builds an actual predictive model for 
30-day readmission risk, while directly testing one of the paper's own methodological choices against an alternative approach, and confirming whether the paper's 
central clinical finding holds up under a different analysis method.

## Dataset

101,766 inpatient diabetes encounters from 130 US hospitals, 1999 to 2008, sourced from the Cerner Health Facts database. Full details in the [original 
paper](http://dx.doi.org/10.1155/2014/781670).

## Methodology

### Data Cleaning

- Dropped `weight` (97% missing) and `payer_code` (40% missing), both too sparse or too tangential to the outcome to be worth imputing
- Kept `medical_specialty` despite 47% missingness, filling gaps with an explicit "Missing" category, since which specialty admitted a patient is plausibly relevant 
to case complexity
- Removed 2,423 encounters ending in death or hospice discharge, since these patients cannot be meaningfully readmitted
- Grouped ICD-9 diagnosis codes into the same broad clinical categories the paper uses (circulatory, respiratory, digestive, diabetes, and so on), validated against 
the paper's own published percentages as a sanity check on the grouping logic

### Handling Repeat Patient Visits

The dataset contains 99,343 encounters from 69,990 unique patients, meaning about 30% of rows are repeat visits. The original paper handles this by keeping only each 
patient's first encounter, which solves the statistical independence problem but discards real information.

This project instead keeps every encounter and handles the dependence problem at the modeling stage using GroupKFold cross-validation, which ensures a single 
patient's visits never end up split across both the training and test sets. This decision was based on evidence that repeat patients carry genuine signal: among 
patients with 2+ visits, personal readmission rates vary widely (median 0%, 75th percentile 33%) rather than clustering around the dataset average, suggesting real 
patient-level risk differences worth preserving.

This decision is tested directly later, not just assumed to be correct.

### Class Imbalance

Only about 11% of encounters are early readmissions. Rather than undersampling (discarding majority-class rows) or oversampling with synthetic data (SMOTE), this 
project uses cost-sensitive weighting, `class_weight='balanced'` for logistic regression and `scale_pos_weight` for XGBoost, which keeps the full dataset intact 
while penalizing misclassifying the minority class more heavily.

### Modeling

A logistic regression baseline was established first, then compared against XGBoost.

| Model | Test ROC-AUC | Notes |
|---|---|---|
| Logistic Regression (baseline) | 0.649 | class-weighted, scaled features |
| XGBoost (untuned) | 0.638 | underperformed the baseline |
| XGBoost (tuned, bug present) | 0.666 | RandomizedSearchCV + GroupKFold |
| **XGBoost (tuned, bug fixed)** | **0.671** | final model |
| CatBoost (untuned, sanity check) | 0.659 | confirms XGBoost result isn't an outlier |
| XGBoost on paper's first-encounter-only data | 0.659 | same model, tests the repeat-visits decision |

**Diagnosing a real bug:** the initially tuned model (0.666) turned out to be treating `admission_type_id`, `discharge_disposition_id`, and `admission_source_id` as 
raw integers rather than categorical codes, meaning the model implicitly assumed, for example, that discharge disposition 20 was numerically "more" of something than 
disposition 5, which isn't what these codes represent. Correcting this with proper one-hot encoding improved test ROC-AUC from 0.666 to 0.671. The gain is modest, 
and that's reported honestly here rather than overstated, but it reflects a genuine bug found and fixed, not just a re-run with different random noise.

**Testing the repeat-visits decision:** running the same tuned model on a rebuilt dataset that matches the paper's first-encounter-only method scores 0.659, 
meaningfully below the 0.671 achieved keeping all encounters with GroupKFold. This confirms the earlier decision to deviate from the paper's approach was the right 
call for a prediction task specifically, even though the paper's own choice was reasonable for their inference-focused goals.

**Benchmarking:** published work on this exact dataset typically reports XGBoost ROC-AUC in the 0.65 to 0.69 range. The original paper reports no predictive accuracy 
at all, its logistic regression was used purely for inference. A 0.671 result sits at or above the honest published benchmark for this dataset, and the dataset's 
known limitations (no post-discharge information, no socioeconomic or care-access data) mean this is close to the realistic ceiling rather than a shortfall in 
modeling effort.

### Model Explainability (SHAP)

SHAP's `TreeExplainer` was used to understand what actually drives the final model's predictions.

**Global feature importance:** prior hospital utilization dominates, `number_inpatient` (inpatient visits in the preceding year) is by far the strongest predictor, 
followed by discharge disposition, age, number of diagnoses, and number of medications. `A1Cresult` itself ranks low in overall importance.

**HbA1c specifically:** despite ranking low globally, the SHAP dependence plot shows a consistent direction, patients who were tested for HbA1c (regardless of 
result) tend toward lower predicted risk than those who weren't tested. This is directionally consistent with the paper's central claim, even though the effect size 
is smaller than prior-utilization features.

### Replicating the Paper's Core Finding

A direct subgroup comparison was run to test the paper's central claim: that HbA1c testing matters more for patients whose primary diagnosis is diabetes itself.

| Group | Not tested | Tested, high result (>8%) | Difference |
|---|---|---|---|
| Diabetes as primary diagnosis | 14.7% readmission | 8.6% readmission | 6.1 points |
| Any other primary diagnosis | 11.4% readmission | 10.4% readmission | 1.0 point |

This replicates the paper's finding directly: HbA1c testing shows a substantially larger association with reduced readmission specifically for patients admitted 
primarily for diabetes, versus a much smaller effect for patients admitted primarily for something else, even when diabetes is a secondary diagnosis for them too.

## Key Takeaways

- Prior hospital utilization is the dominant predictor of 30-day readmission risk in this model, a finding beyond the scope of the original paper
- HbA1c testing has a real but secondary effect, and that effect is concentrated specifically in diabetes-primary admissions, exactly as the original paper found
- Keeping repeat patient visits and handling dependence through GroupKFold measurably outperforms the paper's first-encounter-only approach for a prediction task 
(0.671 vs 0.659 ROC-AUC)
- A real encoding bug was found and fixed mid-project (0.666 to 0.671), a reminder that model performance issues are often data issues, not just algorithm or tuning 
issues

## Repository Structure
clinical-readmission-prediction/
├── data/
│ ├── raw/ # original CSV (gitignored)
│ └── processed/ # cleaned dataset (gitignored)
├── models/
│ └── xgb_readmission_model.pkl
├── notebooks/
│ └── 01_eda.ipynb # full analysis: cleaning, modeling, SHAP
├── reports/
│ └── figures/
│ ├── shap_summary_plot.png
│ └── shap_a1c_dependence_plot.png
├── requirements.txt
└── README.md

## Reproducing This Analysis

1. Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008) and place 
it at `data/raw/diabetic_data.csv`
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/01_eda.ipynb` from top to bottom

## Limitations

This dataset has known ceiling effects for readmission prediction. It captures clinical and demographic information from the hospital encounter itself, but nothing 
about post-discharge circumstances, medication adherence, home support, or socioeconomic factors, all of which meaningfully influence real readmission risk. 
Published work on this dataset consistently lands in a similar performance range regardless of modeling approach, which suggests this ceiling reflects the data's 
limitations rather than a specific modeling shortfall. This project was designed around demonstrating rigorous methodology and honest benchmarking against the 
literature, rather than chasing a headline number.

## Reference

Strack, B., DeShazo, J.P., Gennings, C., Olmo, J.L., Ventura, S., Cios, K.J., & Clore, J.N. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates: 
Analysis of 70,000 Clinical Database Patient Records. *BioMed Research International*, 2014. http://dx.doi.org/10.1155/2014/781670
