# Customer Churn Prediction (Project 9)

## Problem Statement
A telecom company wants to predict which customers are likely to cancel their subscription (churn) based on their service usage, account tenure, and contract details. The objective is to build a machine learning classification model that enables the business to proactively target at-risk customers with retention offers before they leave.

## Additional Requirement
Identify the strongest churn drivers using model coefficients and propose realistic, actionable retention strategies to mitigate customer attrition.

---

## Dataset
- **Dataset Name:** Telco Customer Churn Dataset
- **Source:** Kaggle (blastchar)
- **Size:** 7,043 rows, 21 columns

---

## Tools & Libraries Used
- **Language:** Python 3
- **Data Manipulation:** pandas, numpy
- **Data Visualization:** matplotlib, seaborn
- **Machine Learning:** scikit-learn (LogisticRegression, StandardScaler, train_test_split)

---

## Workflow Checklist

1. **Data Collection:** Loaded dataset into pandas DataFrame.
2. **Data Cleaning:**
   - Converted TotalCharges to numeric, imputing blank whitespace strings (0 tenure) to 0.0.
   - Dropped customerID column.
   - Standardized 'No internet service' and 'No phone service' to 'No'.
   - Checked and dropped duplicate rows.
3. **Exploratory Data Analysis (EDA):**
   - Contract type vs. Churn rate bar plot (Month-to-Month vs 1/2 Year).
   - Tenure distribution KDE plot split by Churn status.
   - Churn rate by Internet Service & Payment Method.
   - Correlation heatmap for numeric features (tenure, MonthlyCharges, TotalCharges).
4. **Feature Engineering:**
   - Categorized tenure into tenure_group (0-12, 13-24, 25-48, 49+ months).
   - Created avg_monthly_spend feature (TotalCharges / tenure).
   - One-hot encoded categorical service features with drop_first=True.
5. **Model Building:**
   - Split dataset into Train (80%) and Test (20%) sets using stratified sampling.
   - Standardized numerical features using StandardScaler.
   - Trained LogisticRegression classifier using class_weight='balanced'.
6. **Evaluation & Interpretation:**
   - Calculated Accuracy, Precision, Recall, and F1-Score.
   - Plotted Confusion Matrix heatmap.
   - Ranked feature coefficients to identify key churn drivers and retention factors.
7. **Actionable Retention Strategies:**
   - Proposed 3 concrete business strategies based on top logistic regression coefficients.

---

## Results & Model Performance

### Evaluation Metrics
- **Model:** Logistic Regression (class_weight='balanced')
- **Accuracy:** ~80%
- **Precision:** ~0.65
- **Recall:** ~0.75 (High recall captures at-risk churners effectively)
- **F1 Score:** ~0.70

### Key Churn Drivers & Retention Drivers

| Top Churn Risk Drivers (Positive Coeff) | Top Retention Drivers (Negative Coeff) |
| :--- | :--- |
| 1. Month-to-Month Contracts (+0.74) | 1. Long-term (1-year & 2-year) Contracts (-0.75) |
| 2. Fiber Optic Internet Service (+0.55) | 2. High Tenure / 49+ months group (-0.58) |
| 3. Electronic Check Payment Method (+0.35) | 3. Online Security & Tech Support Add-ons (-0.38) |
| 4. High Monthly Charges (+0.32) | 4. Auto-pay payment methods (-0.30) |

---

## Actionable Retention Strategies (Additional Requirement)

1. **Contract Transition Incentive (Month-to-Month Mitigation):**
   - *Insight:* Month-to-month contract holders are the strongest predictors of churn.
   - *Strategy:* Target month-to-month customers in their first 6 months with a 15% discount or a free service upgrade if they switch to an annual contract.
2. **Fiber Optic Service Onboarding & Support:**
   - *Insight:* Fiber optic users exhibit higher churn despite paying premium rates, often due to high expectations or service setup friction.
   - *Strategy:* Bundle free 3-month Online Security & Tech Support for new Fiber Optic subscribers and conduct proactive 30-day check-in calls.
3. **Auto-Pay Migration Campaign:**
   - *Insight:* Electronic Check users churn at significantly higher rates compared to automated payment methods.
   - *Strategy:* Offer a one-time 0 account credit for switching from Electronic Check to recurring Auto-Pay (Bank Transfer or Credit Card).

---

## Project Screenshots

- Images/eda_contract_churn.png - Contract Type vs. Churn Rate
- Images/eda_tenure_distribution.png - Tenure Distribution KDE Plot
- Images/eda_correlation_heatmap.png - Feature Correlation Heatmap
- Images/evaluation_confusion_matrix.png - Model Confusion Matrix Heatmap
- Images/churn_driver_coefficients.png - Top Churn Drivers Bar Chart

---

## GitHub Repository Structure

Follow this structure when uploading to GitHub:



---

## Future Improvements
- Test ensemble non-linear models (Random Forest, XGBoost) to compare performance.
- Implement hyperparameter tuning using GridSearchCV.
- Build an interactive web app (Streamlit or Gradio) for real-time churn risk scoring.

---

## Author & Program Details
- **Program:** AIML Industry Readiness Program — Mini ML Project
- **Trainer:** Aditya Dorwal
- **Project:** Project 9 — Customer Churn Prediction

