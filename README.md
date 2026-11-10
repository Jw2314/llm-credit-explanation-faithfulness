# Faithfulness of LLM-Generated Explanations for Credit Risk Decisions

## Research question
When a large language model explains an XGBoost credit-default model's decisions
in plain English, do the explanations faithfully reflect the model's actual
feature attributions (SHAP values)?

## Pre-registered hypotheses (committed before data collection)
- H1: With SHAP values provided (Condition B), the LLM's top-3 cited factors overlap
  with the top-3 SHAP features significantly more than chance (3/16 ≈ 0.19).
- H2: Faithfulness is significantly lower without SHAP values (Condition A)
  than with them (Condition B).
- H3: The LLM makes sign errors (stating the wrong direction of a feature's effect)
  at a measurable rate even in Condition B.

## Data
- Lending Club accepted loans (Kaggle), 36-month term only.
- Train: loans issued 2010–2014. Test: loans issued 2015.
- Outcome: default = "Charged Off"; non-default = "Fully Paid".
- Features: 16 application-time variables (whitelist). Excluded: grade, sub_grade,
  int_rate (outputs of Lending Club's own risk model) and all post-origination fields.

## Methods
- Credit model: XGBoost; logistic regression baseline. Metrics: AUC-ROC, PR-AUC, calibration.
- Attributions: SHAP TreeExplainer (log-odds); one-hot dummies re-aggregated per feature.
- LLM: Qwen/Qwen2.5-7B-Instruct, 4-bit quantised, greedy decoding.
- Sample: 300 test applicants, stratified (clear approve / clear decline / borderline).
- Conditions: A = features + decision; B = features + decision + SHAP values.
- Metrics: top-k overlap, sign accuracy, hallucination rate.

## Statistical tests
- H1: one-sample Wilcoxon signed-rank test vs 0.19.
- H2: paired Wilcoxon signed-rank test, Condition A vs B.
- H3: sign-error rate with bootstrap 95% CI.
- Significance level: 5%, two-sided.

## Status
Pre-registration committed: [today's date]. Data collection not yet started.
