# 🏏 BCCI Cricket Win Prediction — Capstone Project

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **Predicting Indian Cricket Team match wins using historical match data and supervised machine learning — empowering BCCI with data-driven strategic insights.**

---

## Table of Contents
- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Key Results](#key-results)
- [Business Impact](#business-impact)
- [Skills](#skills)
- [Key Learnings](#key-learnings)
- [Future Improvements](#future-improvements)
- [Repository Structure](#repository-structure)
- [Author](#-author)

---

## Project Overview

This project was developed as part of a data analytics consulting engagement for the **Board of Control for Cricket in India (BCCI)**. The goal was to extract actionable insights from historical cricket match data and build machine learning models capable of predicting whether the Indian Cricket Team will **Win** or **Lose** a given match.

Beyond model building, the project also delivers **unique, match-specific strategies** for India's upcoming fixtures across different opponents, formats, and conditions — making this a practical, business-ready solution rather than a purely academic exercise.

👉 [Open the notebook to explore full analysis](notebook/BCCI_Cricket_Win_Prediction.ipynb)

---

## Business Problem

The **BCCI** engaged an external analytics consulting firm to leverage historical match data for strategic decision-making. The primary objectives were:

- Build machine learning classification models to accurately predict a **Win for Team India**
- Extract **actionable recommendations** from model outputs and EDA insights
- Predict results for **5 upcoming matches** (Test vs England in England, T20 vs Australia in India, ODI vs Sri Lanka in India) and suggest **unique, feasible winning strategies** for each — ensuring no strategy is repeated across the series

The stakeholders include cricket strategists, team selectors, and management officials at BCCI. The decisions informed by this model can directly influence **team composition**, **match tactics**, and **venue preparation strategies**.

---

## Dataset

| Attribute | Detail |
|---|---|
| **Source** | Internal BCCI historical match records (`Sports Data.xlsx`) |
| **Size** | 2,930 rows × 23 columns |
| **Target Variable** | `Result` (Win / Loss) |
| **Class Distribution** | ~83.9% Win, ~16.1% Loss |
| **Numeric Features** | 13 columns (float64 + int64) |
| **Categorical Features** | 10 columns (object) |

**Key Features include:**
`Avg_team_Age`, `Match_light_type`, `Match_format`, `Bowlers_in_team`, `All_rounder_in_team`, `Wicket_keeper_in_team`, `First_selection`, `Opponent`, `Season`, `Audience_number`, `Offshore`, `Max_run_scored_1over`, `Max_wicket_taken_1over`, `Extra_bowls_bowled`, `Min_run_given_1over`, `Min_run_scored_1over`, `Max_run_given_1over`, `extra_bowls_opponent`, `player_highest_run`, `Players_scored_zero`, `player_highest_wicket`

---

## Methodology

### 1. Data Understanding

The dataset was loaded from Google Drive using Pandas. Initial exploration included `.head()`, `.tail()`, `.shape`, `.info()`, and `.describe()` to understand the structure and statistical profile. The dataset contains **2,930 records** across **23 columns**, with a mix of float, integer, and object data types. No duplicate records were found.

### 2. Data Cleaning & Preprocessing

**Missing Value Treatment:** A total of 789 missing values were identified across 15 columns, with `Avg_team_Age` (3.31%) and `Bowlers_in_team` (2.80%) having the highest missing percentages — all within acceptable limits. Categorical columns were imputed using **mode**, and numerical columns using **mean**.

**Data Type Corrections:** Columns `Players_scored_zero` and `player_highest_wicket` contained mixed types (numeric values stored as strings — e.g., `'Three'` instead of `3`). These were corrected using `.replace()`. The `Match_format` column had an inconsistent label `'20-20'` which was standardized to `'T20'`. All float columns were subsequently converted to `int64` for modeling consistency.

**Outlier Treatment:** Boxplots revealed outliers primarily in `Avg_team_Age`. IQR-based capping was applied to this column by replacing values beyond the lower and upper bounds with the respective boundary values. Other numeric columns showed natural spread and were retained as-is.

### 3. Exploratory Data Analysis (EDA)

**Univariate Analysis:** Distribution plots (histograms with KDE) were generated for all numeric columns, and count plots were created for all categorical features. Key observations include a heavily right-skewed distribution for `Audience_number`, near-uniform distribution for `player_highest_run`, and class dominance in `Result` (Win majority).

**Bivariate Analysis:** Correlation heatmaps (Seaborn `coolwarm` palette) were plotted for numerical features, with `Wicket_keeper_in_team` excluded due to zero variance. Stacked count plots revealed the relationship between categorical features (Match_light_type, Match_format, First_selection, Offshore) and the target `Result`. Box plots compared the distribution of numeric features across Win and Loss outcomes.

**Targeted Bivariate Explorations:**
- *All-Rounders vs Max Runs in 1 Over:* The trend line was nearly flat, indicating no significant correlation — specialist bowlers may be more impactful.
- *Season vs Audience Attendance:* Rainy season had the most audience variability, with a notable outlier exceeding 1.4 million.
- *Bowlers in Team vs Max Run Given:* Higher bowler rotation (3–4 bowlers) corresponded with more frequent big overs, possibly exposing weaker links.
- *Avg Team Age vs Players Scoring Zero:* A slight negative trend indicated younger teams may have more ducks, though the relationship was weak.
- *All-Rounders vs Extras Bowled:* Teams with 3 all-rounders bowled the fewest extras on average, suggesting a disciplined sweet spot.
- *Match Light Type vs Max Run Scored:* Night matches showed slightly higher median scores, potentially due to dew or better batting conditions under lights.

**EDA Insights Summary:**
- Teams with **3–4 bowlers and at least 2 all-rounders** tend to win more often.
- **Home matches** (non-offshore) showed significantly higher win percentages.
- **Day & Night format** and **batting first** slightly favored India.
- Higher **audience attendance** correlated with match wins, reflecting crowd-driven uplift.
- Winning teams tend to score **20+ runs in at least one over**, feature a **top scorer above 75**, and **bowl fewer extras**.

### 4. Data Preparation for Modeling

- The `Game_number` column was dropped as it is a non-predictive identifier.
- The target variable `Result` was label-encoded: `Loss → 0`, `Win → 1`.
- Categorical features were one-hot encoded using `pd.get_dummies(drop_first=True)`.
- A constant term was added via `statsmodels.api.add_constant()` for Logistic Regression.
- Data was split into **80% train / 20% test** using `train_test_split` with `random_state=1`. Class proportions were verified to be consistent across splits (~83.9% Win in both).

### 5. Model Building

Four classification algorithms were trained and evaluated:

**Logistic Regression (Statsmodels + Sklearn):** An initial logistic regression was built using Statsmodels to inspect p-values and coefficients. Multicollinearity was assessed using **Variance Inflation Factor (VIF)**, and features with VIF > 5 (`First_selection_Batting`, `Opponent_South Africa`, `Match_format_T20`, `player_highest_wicket`) were dropped iteratively. Remaining high p-value features were removed using a backward elimination loop. An **ROC curve** was plotted to determine the optimal classification threshold (0.847), which was then used for final predictions.

**Naive Bayes (GaussianNB):** Features were scaled using `StandardScaler` before training the Gaussian Naive Bayes model. This model served as a probabilistic baseline.

**K-Nearest Neighbors (KNN):** An initial KNN was trained with `k=3`. To optimize, k values from 2 to 20 were iterated, with **precision score** used as the selection metric. The best k was found to be **k=2** (precision: 0.9756).

**Decision Tree Classifier:** An initial unpruned Decision Tree was built with `random_state=42`, resulting in perfect training scores (1.000) — a clear sign of overfitting. Pre-pruning was applied using **GridSearchCV** with 5-fold cross-validation across parameters: `max_depth`, `max_leaf_nodes`, `min_samples_split`, and `class_weight`. The best estimator was `max_depth=7, max_leaf_nodes=75, min_samples_split=5`.

### 6. Evaluation

All models were evaluated using **Accuracy, Recall, Precision, and F1-Score** on both training and test sets, along with **confusion matrices**. The Tuned Decision Tree was additionally visualized as a tree diagram and a text rule report was exported. Feature importances were ranked and plotted.

---

## Key Results

### Model Performance Summary

| Model | Test Accuracy | Test Recall | Test Precision | Test F1 |
|---|---|---|---|---|
| Logistic Regression (Base) | 0.872 | 0.967 | 0.890 | 0.927 |
| Logistic Regression (Tuned) | 0.729 | 0.721 | 0.941 | 0.817 |
| Naive Bayes (Base) | 0.584 | 0.536 | 0.943 | 0.683 |
| KNN (Base, k=3) | 0.846 | 0.945 | 0.880 | 0.912 |
| KNN (Tuned, k=2) | 0.894 | 0.896 | 0.976 | 0.934 |
| Decision Tree (Base) | 0.935 | 0.947 | 0.975 | 0.961 |
| **Decision Tree (Tuned) ✅** | **0.874** | **0.955** | **0.900** | **0.927** |

### Top Feature Importances (Tuned Decision Tree)

1. `Audience_number` — Most predictive feature; match popularity has strong signal
2. `player_highest_run` — Individual batting performance is a key win indicator
3. `Offshore_Yes` — Playing away has a strong negative effect on win probability
4. `All_rounder_in_team` — Balanced team composition drives wins
5. `extra_bowls_opponent` and `Extra_bowls_bowled` — Bowling discipline matters for both sides

### Selected Model: Tuned Decision Tree Classifier

The **Tuned Decision Tree** was selected as the final production model due to:
- Best balance between recall (0.955) and generalization (no overfitting after pruning)
- High interpretability — tree rules can be directly translated into playing strategies
- Consistent train-test performance gap (0.927 train vs 0.874 test accuracy), indicating well-controlled variance
- Strong F1-score ensuring minimal missed wins in predictions

---

## Business Impact

**1. Data-Driven Team Selection**
Teams with 3–4 specialist bowlers, at least 2 all-rounders, and a fixed wicket-keeper should be prioritized. Player selection committees can use the model's feature importances as a quantitative checklist rather than relying solely on intuition.

**2. Match-Specific Strategic Planning**
For each upcoming match (Test vs England in rainy conditions, T20 vs Australia in Indian winter, ODI vs Sri Lanka in Indian winter), unique strategies were derived using the model to ensure predicted wins while avoiding strategic repetition detectable by opponents.

**3. Home Advantage Maximization**
The `Offshore_Yes` feature ranks as one of the most predictive for loss. BCCI should maximize home fixtures and, when playing abroad, invest heavily in training under similar conditions to neutralize this disadvantage.

**4. Performance Monitoring Framework**
Integrating the trained model into the pre-match analytics pipeline can provide win probability scores in real time, allowing coaching staff to make last-minute XI changes with quantified confidence.

**5. Fan Engagement & Commercial Strategy**
Higher `Audience_number` strongly correlates with wins. BCCI can use this to justify prioritizing high-attendance venues for critical series and to align broadcast and sponsorship strategy around matches with the greatest commercial upside and highest predicted win probability.

---

## Skills

### Technical Skills
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-77AC1D?style=for-the-badge&logo=seaborn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-000000?style=for-the-badge&logo=matplotlib&logoColor=white)
![Statsmodels](https://img.shields.io/badge/Statsmodels-4B0082?style=for-the-badge&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Exploratory Data Analysis](https://img.shields.io/badge/Data_Analysis-FFA500?style=for-the-badge&logo=google-analytics&logoColor=white)
![Google Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)

### Soft Skills
![Analytical Thinking](https://img.shields.io/badge/Analytical_Thinking-4B0082?style=for-the-badge&logo=mindmap&logoColor=white)
![Communication](https://img.shields.io/badge/Communication-25D366?style=for-the-badge&logo=google-messages&logoColor=white)
![Problem Solving](https://img.shields.io/badge/Problem_Solving-FF4500?style=for-the-badge&logo=brainly&logoColor=white)
![Attention to Detail](https://img.shields.io/badge/Attention_to_Detail-00CED1?style=for-the-badge&logo=google-search-console&logoColor=white)

---

## Key Learnings

- **Class imbalance awareness:** With ~84% wins in the dataset, accuracy alone is a misleading metric — recall and F1 are far more meaningful for evaluating model fitness for this problem.
- **VIF-based feature selection** before logistic regression is essential for producing statistically reliable coefficients and p-values; skipping this step inflates standard errors and misleads interpretation.
- **Threshold optimization via ROC curve** significantly shifts the precision-recall trade-off. The optimal threshold of 0.847 (vs the default 0.5) dramatically changed prediction behavior for the tuned logistic model.
- **Decision Tree interpretability** is a genuine business asset — the exported tree rules can be read as playing strategies, not just classification outputs.
- **GridSearchCV with recall as the scoring metric** is the right approach when false negatives (missed wins in predictions) are more costly than false positives.
- **Domain context matters:** Features like `Offshore_Yes`, `Season`, and `Audience_number` seem auxiliary but turn out to be among the most predictive — statistical intuition must be grounded in cricket knowledge to be useful.

---

## Future Improvements

1. **Ensemble Methods:** Explore Random Forest, Gradient Boosting (XGBoost, LightGBM) for potentially superior performance and built-in feature importance.
2. **Class Imbalance Handling:** Apply SMOTE or class-weighted models to improve minority class (Loss) recall, which is critical for risk-aware match strategy planning.
3. **Advanced Feature Engineering:** Create derived features such as run-rate differentials, innings-specific performance ratios, and cumulative player form scores to enrich the feature space.
4. **Real-Time Prediction Pipeline:** Integrate the model into a live data pipeline that ingests pre-match data (weather, venue stats, squad announcements) and outputs win probability scores with confidence intervals.
5. **Opponent-Specific Sub-Models:** Given that performance varies significantly by opponent (West Indies had the highest positive coefficient in logistic regression), training separate or stratified models per opponent could yield more granular and actionable predictions.

---

## Repository Structure

```
cricket-win-prediction-bcci-ml/
│
├── data/
│   ├── Comp_Fin_Data.csv                             # Corporate financial dataset (Part A)
│   └── Market_Risk_Data.csv                          # Stock price dataset (Part B)
│
├── notebook/
│   └── finance-retail-analytics-using-python.ipynb   # Analysis notebook
│
├── README.md                                         # Project documentation
├── LICENSE                                           # License file
└── .gitignore                                        # Git ignore file
```
---

## 👨‍💻 Author

**Nabankur Ray**

Passionate about real-world data-driven solutions

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?style=flat&logo=github)](https://github.com/nabankur14) [![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/nabankur-ray-876582181/)

![GitHub Stats](https://github-readme-stats-eight-theta.vercel.app/api?username=nabankur14&show_icons=true)

⭐ If you like this project

Give it a ⭐ on GitHub — it helps a lot!
