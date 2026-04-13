# 💳 Credit Scoring — Default Risk Prediction

> Machine Learning project for predicting the probability of financial default using the **GiveMeSomeCredit** dataset.

## 🚀 Live Demo

👉 **[Credit Scoring Dashboard — Click to open](https://mlproject20-mz8ohmrciayhzf69xnbp6k.streamlit.app/)**

> Interactive app with real-time prediction + SHAP explainability

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Streamlit Application](#-streamlit-application)
- [Technologies](#-technologies)
- [How to Run Locally](#-how-to-run-locally)
- [Author](#-author)

---

## 📌 Project Overview

Credit scoring is a critical task in the banking and financial industry. This project builds a complete Machine Learning pipeline to **predict whether a borrower will experience serious financial distress within 2 years**.

The pipeline covers:
- Exploratory Data Analysis (EDA)
- Data Preprocessing & Feature Engineering
- Model Training & Comparison (4 baseline models)
- Hyperparameter Optimization (GridSearchCV)
- Model Explainability (SHAP)
- Deployment as an interactive Streamlit web application

---

## 📊 Dataset

| Property | Details |
|---|---|
| **Name** | GiveMeSomeCredit |
| **Source** | Kaggle Competition |
| **Samples** | 150,000 clients |
| **Features** | 10 financial & demographic variables |
| **Target** | `SeriousDlqin2yrs` (1 = default, 0 = no default) |
| **Class Imbalance** | ~14:1 (Good Payers vs Defaults) |

### Features Description

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Total balance on credit cards / credit limits |
| `age` | Age of the borrower |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times the borrower was 30-59 days late |
| `DebtRatio` | Monthly debt payments / monthly income |
| `MonthlyIncome` | Monthly income of the borrower |
| `NumberOfOpenCreditLinesAndLoans` | Number of open loans and credit lines |
| `NumberOfTimes90DaysLate` | Times the borrower was 90+ days late |
| `NumberRealEstateLoansOrLines` | Number of mortgage and real estate loans |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times the borrower was 60-89 days late |
| `NumberOfDependents` | Number of dependents in the family |

---

## 📁 Project Structure

```
ML_Project_20/
│
├── 📓 Credit_Scoring_DRAFT.ipynb         # Draft 1 — Initial exploration & baseline models
├── 📓 Credit_Scoring_DRAFT_V2.ipynb      # Draft 2 — XGBoost hyperparameter tuning
├── 📓 Credit_Scoring_Version_Finale.ipynb # Final version — Clean & complete pipeline
│
├── 🌐 app.py                             # Streamlit web application
├── 📦 best_model.pkl                     # Saved champion model (XGBoost Optimized)
├── 📦 scaler.pkl                         # Saved StandardScaler
├── 📄 requirements.txt                   # Python dependencies
├── 📊 GiveMeSomeCredit.csv               # Dataset
└── 📖 README.md                          # Project documentation
```

---

## 🔬 Methodology

### 1. Exploratory Data Analysis (EDA)
- Class distribution analysis → severe imbalance detected (14:1)
- Missing values analysis → `MonthlyIncome` (19.82%), `NumberOfDependents` (2.62%)
- Outlier detection and visualization

### 2. Data Preprocessing
- **Imputation** : Missing values replaced by median
- **Outlier Capping** : Winsorization at 99th percentile
- **Train/Test Split** : 80% / 20% with stratification
- **SMOTE** : Synthetic Minority Over-sampling to handle class imbalance
  - Before SMOTE : 111,979 vs 8,021 samples
  - After SMOTE : 111,979 vs 111,979 samples (balanced)
- **Normalization** : StandardScaler applied for distance-based models

### 3. Model Training & Comparison

4 baseline models were trained and evaluated:

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8451 | 0.3002 | 0.7641 |
| Random Forest | 0.8215 | 0.3484 | 0.8924 |
| XGBoost | 0.8083 | 0.3292 | 0.8742 |
| KNN | 0.7782 | 0.2797 | 0.7813 |

### 4. Hyperparameter Optimization
- **Algorithm** : GridSearchCV with 3-fold cross-validation
- **Model** : XGBoost Classifier
- **Search space** : `n_estimators`, `max_depth`, `learning_rate`, `subsample`
- **Best parameters** : `learning_rate=0.1`, `max_depth=7`, `n_estimators=200`, `subsample=0.8`

### 5. Final Comparison (All 5 Models)

| Model | AUC-ROC | F1-Score | Accuracy |
|---|---|---|---|
| Logistic Regression | 0.8451 | 0.3002 | 0.7641 |
| Random Forest | 0.8215 | 0.3484 | 0.8924 |
| XGBoost | 0.8083 | 0.3292 | 0.8742 |
| KNN | 0.7782 | 0.2797 | 0.7813 |
| **XGBoost (Optimized)** | **0.9532** | **—** | **—** |

---

## 🏆 Results

**Champion Model : XGBoost (Optimized)**

| Metric | Score |
|---|---|
| **AUC-ROC** | **0.9532** |
| Best CV Score | 0.9532 |
| Optimization | GridSearchCV (72 fits) |

> AUC-ROC was chosen as the primary metric because the dataset is highly imbalanced. A high AUC means the model correctly separates good payers from defaulters regardless of the classification threshold.

---

## 🌐 Streamlit Application

The deployed app includes:

### 🎯 Tab 1 — Prediction
- Interactive sidebar form with all 10 client features
- Real-time default risk prediction
- Visual risk gauge (0% → 100%)
- Metric cards : Default Probability, Risk Level, Monthly Income, Age
- Input summary table

### 🔬 Tab 2 — SHAP Explanation
- Feature contribution bar chart (red = increases risk, green = decreases risk)
- Top 3 risk factors with SHAP values
- Compatible with all model types (TreeExplainer / LinearExplainer)

👉 **[Open the app](https://mlproject20-mz8ohmrciayhzf69xnbp6k.streamlit.app/)**

---

## 🛠️ Technologies

| Category | Tools |
|---|---|
| **Language** | Python 3.10+ |
| **Data** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Imbalanced Data** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP |
| **Deployment** | Streamlit, Streamlit Cloud |
| **Versioning** | Git, GitHub |

---

## ▶️ How to Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/Raedaloauni/ML_Project_20.git
cd ML_Project_20

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 👨‍💻 Author

**Raed Aloauni**
- 🎓 Software Engineering Student — École Polytechnique de Sousse
- 🐙 GitHub : [@Raedaloauni](https://github.com/Raedaloauni)

---

*Project realized as part of the Machine Learning module — Academic Year 2024/2025*
