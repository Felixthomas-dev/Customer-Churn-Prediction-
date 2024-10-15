# Customer-Churn-Prediction
![Customer Churn](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7GUbX4KS8wKcH_BufZv0IDYxJoGfeDoMLrQ&s)  <!-- https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS7GUbX4KS8wKcH_BufZv0IDYxJoGfeDoMLrQ&s -->

The Customer Churn Prediction Model for TeleServe is designed to identify customers at risk of leaving the service. Utilizing machine learning algorithms, primarily Logistic Regression, the model analyzes historical customer data, including demographics, contract types, service usage, and payment history.

**A Data-Driven Approach to Reducing Customer Attrition**

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset Information](#dataset-information)
5. [Project Workflow](#project-workflow)
6. [Data Preprocessing](#data-preprocessing)
7. [Modeling and Evaluation](#modeling-and-evaluation)
8. [Key Insights](#key-insights)
9. [Business Impact](#business-impact)
10. [Recommendations & Next Steps](#recommendations--next-steps)
11. [How to Run](#how-to-run)
12. [Technologies Used](#technologies-used)

---

## Project Overview

This project focuses on predicting customer churn for **TeleServe**, a leading global telecommunications company. The aim is to build a model that identifies customers at risk of leaving, enabling the company to implement proactive retention strategies.

## Problem Statement

TeleServe is experiencing high customer churn rates, particularly among customers with short tenure and those on month-to-month contracts. The challenge is to predict customer churn accurately and reduce attrition by 5% within a year.

## Objectives

- **Develop a churn prediction model** to identify at-risk customers early.
- Implement targeted interventions to reduce churn.
- Provide actionable insights into factors influencing churn.

## Dataset Information

- **Records:** 7043 customer records
- **Features:** Customer demographics, contract types, service usage, and charges
- **Target Variable:** `Churn` (1 = At-risk, 0 = Not-at-risk)

## Project Workflow

1. **Business Understanding**: Define goals and objectives.
2. **Data Collection**: Gather data from CRM, including customer contracts, payments, and demographics.
3. **Data Cleaning**: Handle missing values, outliers, and duplicates.
4. **Exploratory Data Analysis (EDA)**: Derive insights on customer behavior and patterns.
5. **Data Preprocessing**: Feature selection, encoding, and scaling.
6. **Model Training**: Implement machine learning models (Random Forest, Logistic Regression, etc.).
7. **Model Evaluation**: Evaluate models and optimize for performance.
8. **Model Deployment**: Deploy the best model into the CRM system for real-time churn prediction.

## Data Preprocessing

- **Handling Missing Data**: Missing values were filled or removed where necessary.
- **Scaling**: Numerical variables such as `Monthly Charges` and `Total Charges` were standardized.
- **Encoding**: Categorical variables like `Contract Type` and `Gender` were converted into numerical form.
- **Class Imbalance**: Addressed using techniques like oversampling to balance churn vs. non-churn customers.
- **Train-Test Split**: The data was split into 80% training and 20% testing.

## Modeling and Evaluation

Multiple machine learning models were used, including:

- **Logistic Regression**: Best performing model with a recall of 0.58 and an accuracy of 81.69%.
- **Random Forest**: Achieved 80.34% accuracy, though with lower recall for class 1 (churn).
- **AdaBoost**, **XGBoost**, **SVC**, and **Decision Trees** were also tested.

| Model                | Accuracy | AUC-ROC | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) |
|----------------------|----------|---------|---------------------|------------------|-------------------|
| Logistic Regression   | 81.69%   | 0.8593  | 0.68                | 0.58             | 0.63              |
| AdaBoost              | 80.91%   | 0.8580  | 0.68                | 0.54             | 0.60              |
| RandomForest          | 80.34%   | 0.8578  | 0.67                | 0.50             | 0.58              |
| SVC                   | 80.62%   | 0.8088  | 0.68                | 0.50             | 0.58              |
| XGBoost               | 79.13%   | 0.8365  | 0.63                | 0.51             | 0.57              |
| Decision Tree         | 71.97%   | 0.6456  | 0.47                | 0.49             | 0.48              |

## Key Insights

- **High Churn Rate**: The overall churn rate is 27%.
- **Tenure Impact**: Customers with less than 12 months of tenure are most likely to churn.
- **Contract Type**: Month-to-month contract customers have the highest churn rate at 23%.
- **Service Type**: Customers using fiber optic internet are at higher risk (18% churn rate).
- **Demographics**: Customers without dependents or a partner are more likely to churn.

## Business Impact

- **High Model Accuracy**: The 81.7% accurate model enables TeleServe to focus retention efforts on high-risk customers.
- **Cost Savings**: A projected 5% reduction in churn can save the company approximately $1.5 million annually.
- **Targeted Campaigns**: The model informs personalized outreach strategies, improving customer engagement.

## Recommendations & Next Steps

- **Tenure-Based Offers**: Provide incentives for customers in their first 12 months to reduce churn.
- **Service Personalization**: Offer flexible plans for customers with high charges.
- **Real-Time CRM Alerts**: Integrate churn predictions into the CRM to enable timely interventions.
- **Model Monitoring**: Continuously update and optimize the model to ensure it adapts to changing customer behaviors.

## How to Run

1. Clone the repository:
    ```bash
    git clone https://github.com/Felixthomas-dev/Customer-Churn-Prediction-
    ```
2. Install required libraries:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook:
    ```bash
    jupyter notebook Customers-Churn_capstone_project.ipynb
    ```

## Technologies Used

- **Python**: Data analysis and model building
- **Scikit-learn**: Machine learning models
- **Pandas & NumPy**: Data manipulation and preprocessing
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Development environment
- **GitHub**: Version control
