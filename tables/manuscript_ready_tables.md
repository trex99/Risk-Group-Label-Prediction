# Manuscript-ready tables

Table metrics are outer-fold means. Top-k metrics use concatenated OOF or sealed holdout predictions.
All final ensemble probabilities come from VotingClassifier.predict_proba().

## Table 1. Composition of Analysis Data

| Category | Rows | Main Variables | Description |
|---|---|---|---|
| Test-level records | 944,767 | Test_id, Test, Label | Test-level training data with risk-group label |
| New Test (A) details | 647,241 | PrimaryKey, Age, TestDate, A1~A9 | Detailed results of the New Test (A) |
| Qualification-Maintenance Test (B) details | 297,526 | PrimaryKey, Age, TestDate, B1~B10 | Detailed results of the Qualification-Maintenance Test (B) |
| Class balance | 2.8877% | Risk group | Low positive rate; imbalanced binary prediction task |

## Table 2. Feature Groups Used in Model

| Feature Group | Representative Variables | Interpretation |
|---|---|---|
| Age and timing | Age, TestDate_year, TestDate_month, YearMonthIndex | Risk-group patterns by age band and test timing |
| Past history | prev_ab_all_label_mean, prev_all_label_mean | Same worker's past tests and risk-group history |
| Cross history | other_test_Test_id_count, other_test_Label_mean | Repeated-testing and cross-test information between A and B |
| Accuracy and error statistics | acc_stats_*, err_stats_* | Level and variability of correct/incorrect performance |
| Response-time statistics | rt_mean_stats_*, rt_std_stats_* | Response speed and response stability |
| Missing-count feature | isna_sum | Number of missing final features after fold-specific history reconstruction |

## Table 3. Performance of Baselines, Single Models and VotingClassifier

| Model | Score | AUC | PR-AUC | Brier | ECE (uniform-10) | Adaptive ECE (quantile-10) |
|---|---|---|---|---|---|---|
| DummyClassifier | 0.257012 | 0.500000 | 0.028877 | 0.028043 | 0.000003 | 0.013625 |
| LogisticRegression | 0.161806 | 0.690003 | 0.118645 | 0.026769 | 0.000459 | 0.002219 |
| HistGradientBoosting | 0.147118 | 0.719082 | 0.156478 | 0.026162 | 0.000475 | 0.001053 |
| LightGBM | 0.147059 | 0.719222 | 0.157305 | 0.026145 | 0.000538 | 0.001188 |
| XGBoost | 0.147037 | 0.719264 | 0.157108 | 0.026148 | 0.000527 | 0.001072 |
| CatBoost | 0.147130 | 0.719017 | 0.158147 | 0.026114 | 0.000442 | 0.001190 |
| Soft-voting VotingClassifier | 0.146790 | 0.719737 | 0.158234 | 0.026126 | 0.000509 | 0.001181 |

## Table 4. Feature-Group Ablation

| Condition | Features | Score | ΔScore | AUC | PR-AUC |
|---|---|---|---|---|---|
| All features | 37 | 0.146790 | 0.000000 | 0.719737 | 0.158234 |
| Remove accuracy/error statistics | 29 | 0.147496 | 0.000706 | 0.718325 | 0.158119 |
| Remove age/timing | 33 | 0.168572 | 0.021782 | 0.676417 | 0.125051 |
| Remove past/cross-test history features | 21 | 0.161845 | 0.015055 | 0.690226 | 0.067764 |
| Remove missing-count feature | 36 | 0.146804 | 0.000014 | 0.719714 | 0.158266 |
| Remove response-time statistics | 29 | 0.146750 | -0.000040 | 0.719815 | 0.158067 |

## Table 5. Stepwise Feature Sets

| Model | Feature set | Score | AUC | PR-AUC |
|---|---|---|---|---|
| A | Age/timing only | 0.171934 | 0.670020 | 0.054685 |
| B | Age/timing + history | 0.147720 | 0.717899 | 0.155141 |
| C | Current cognitive/response summaries | 0.191116 | 0.631724 | 0.044588 |
| D | Full features | 0.146790 | 0.719737 | 0.158234 |

## Table 6. Incremental-Value Cluster Bootstrap

| Metric | Observed | Bootstrap Mean | 95% CI |
|---|---|---|---|
| ΔAUC (Model D − Model B) | 0.001935 | 0.001913 | 0.001194 ~ 0.002596 |
| ΔPR-AUC (Model D − Model B) | 0.002823 | 0.002845 | 0.002082 ~ 0.003634 |

## Table 7. Sensitivity Analysis of Prior Test History and PrimaryKey Separation

| Condition | History Features | Features | Score | AUC | PR-AUC |
|---|---|---|---|---|---|
| Main nested CV | Included | 37 | 0.146790 | 0.719737 | 0.158234 |
| Main outer folds | Excluded | 21 | 0.169473 | 0.674953 | 0.056860 |
| PrimaryKey-disjoint CV | Excluded | 21 | 0.169843 | 0.674315 | 0.056717 |

## Table 8. Nested OOF Top-k

| Top-k | n | Label Count | Label Rate | Lift | Cumulative Recall | Overall Label Rate |
|---|---|---|---|---|---|---|
| Top 1% | 9,448 | 3,405 | 36.04% | 12.48 | 12.48% | 2.8877% |
| Top 5% | 47,239 | 6,561 | 13.89% | 4.81 | 24.05% | 2.8877% |
| Top 10% | 94,477 | 9,502 | 10.06% | 3.48 | 34.83% | 2.8877% |

## Table 9. Time-Specification Robustness

| Time specification | Nested-CV AUC | Nested-CV PR-AUC | Temporal AUC | Temporal PR-AUC |
|---|---|---|---|---|
| Full absolute-time features | 0.719737 | 0.158234 | 0.692449 | 0.093923 |
| Remove current absolute-time variables | 0.700138 | 0.132661 | 0.650682 | 0.084915 |
| Relative-time histories | 0.703427 | 0.151804 | 0.694232 | 0.107706 |

## Table 10. Temporal Holdout Top-k

| Condition | Top-k | Label Rate | Lift | Cumulative Recall |
|---|---|---|---|---|
| Full absolute-time features | 1% | 27.00% | 10.61 | 10.61% |
| Full absolute-time features | 5% | 11.37% | 4.47 | 22.34% |
| Full absolute-time features | 10% | 7.73% | 3.04 | 30.38% |
| Remove current absolute-time variables | 1% | 23.75% | 9.33 | 9.33% |
| Remove current absolute-time variables | 5% | 10.96% | 4.30 | 21.52% |
| Remove current absolute-time variables | 10% | 7.32% | 2.88 | 28.76% |
| Relative-time histories | 1% | 27.49% | 10.80 | 10.80% |
| Relative-time histories | 5% | 11.07% | 4.35 | 21.73% |
| Relative-time histories | 10% | 7.66% | 3.01 | 30.09% |

## Table 11. Test-Type Subgroup Performance

| Condition | New Test (A) AUC | New Test (A) PR-AUC | Qualification-Maintenance Test (B) AUC | Qualification-Maintenance Test (B) PR-AUC |
|---|---|---|---|---|
| Nested OOF full model | 0.706512 | 0.117652 | 0.696542 | 0.195147 |
| Nested OOF Model B | 0.703608 | 0.115906 | 0.693683 | 0.191991 |
| Nested OOF Model C | 0.622886 | 0.035709 | 0.550172 | 0.050566 |
| Temporal full features | 0.662214 | 0.050781 | 0.668744 | 0.125223 |
| Temporal no absolute time | 0.614282 | 0.045939 | 0.652272 | 0.117816 |
| Temporal relative time | 0.659710 | 0.053371 | 0.654264 | 0.145363 |
