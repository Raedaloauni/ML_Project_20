#  Credit Scoring & Risk Prediction Project

##  Project Overview
This project aims to predict the probability of financial distress for bank customers. It demonstrates a complete Machine Learning pipeline, from data preprocessing to advanced model optimization.

##  Project Structure & Evolution
To show the progress of the analysis, I have included different versions of the work:

1.  **[Credit_Scoring_DRAFT_V1.ipynb](./Credit_Scoring_DRAFT_V1.ipynb)**: Initial exploration.
    * *Observation:* Simple models like Logistic Regression (AUC 0.845) outperformed default complex models.
2.  **[Credit_Scoring_DRAFT_V2.ipynb](./Credit_Scoring_DRAFT_V2.ipynb)**: Model Optimization.
    * *Action:* Applied `GridSearchCV` to fine-tune XGBoost.
    * *Result:* **XGBoost AUC increased from 0.80 to 0.95**, becoming the new Champion Model.
3.  **[Credit_Scoring_FINAL.ipynb](./Credit_Scoring_FINAL.ipynb)**: Final production-ready report.



##  Tech Stack
* **Language:** Python
* **Environment:** Google Colab / GitHub
* **Libraries:** Scikit-Learn, XGBoost, Pandas, Matplotlib, Seaborn

##  Saved Assets
* `best_model.pkl`: The optimized XGBoost model.
* `scaler.pkl`: The StandardScaler for data normalization.
