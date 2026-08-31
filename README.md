# 💳 Credit Card Fraud Detection

An end-to-end Machine Learning project that detects potentially fraudulent credit card transactions using **XGBoost**, with **SMOTE for class imbalance**, **StandardScaler for numerical features**, and a **Streamlit web application** for real-time prediction.

## 🚀 Project Overview

Credit card fraud detection is a highly imbalanced classification problem where fraudulent transactions are much fewer than legitimate transactions.

The goal of this project is to build a machine learning model that can identify suspicious transactions while maintaining a good balance between **precision and recall**.

The final solution uses **XGBoost** with an optimized classification threshold of **0.4**.

## 🎯 Objectives

* Analyze transaction data and identify fraud patterns
* Perform data preprocessing and feature engineering
* Handle class imbalance using SMOTE
* Scale numerical features using StandardScaler
* Train and compare multiple machine learning models
* Optimize the classification threshold
* Select the best-performing model
* Save the trained model for deployment
* Build an interactive Streamlit application
* Deploy the application for real-time fraud prediction

## 🛠️ Technologies Used

### Programming Language

* Python

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-learn
* XGBoost
* Imbalanced-learn (SMOTE)

### Model Deployment

* Streamlit
* Joblib

### Development Tools

* Jupyter Notebook / Google Colab
* VS Code
* Git
* GitHub

## 📊 Machine Learning Workflow

```text
Raw Transaction Data
        ↓
Data Cleaning
        ↓
Exploratory Data Analysis
        ↓
Feature Engineering
        ↓
Train-Test Split
        ↓
StandardScaler
        ↓
SMOTE on Training Data
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Threshold Optimization
        ↓
Final XGBoost Model
        ↓
Model Serialization
        ↓
Streamlit Application
        ↓
Deployment
```

## ⚖️ Handling Class Imbalance

Fraudulent transactions represent a minority of the dataset.

To address this problem, **SMOTE (Synthetic Minority Oversampling Technique)** was applied only to the training data.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

x_train_smote, y_train_smote = smote.fit_resample(
    x_train,
    y_train
)
```

The test data was kept untouched so that model performance could be evaluated on data representing the original distribution.

## 📏 Feature Scaling

The following numerical features were standardized using `StandardScaler`:

* `amount`
* `avg_transaction_amount`
* `hour`

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_train[num_cols] = scaler.fit_transform(
    x_train[num_cols]
)

x_test[num_cols] = scaler.transform(
    x_test[num_cols]
)
```

The scaler was fitted only on the training data to prevent data leakage.

## 🤖 Models Compared

Three classification algorithms were evaluated:

1. Logistic Regression
2. Random Forest
3. XGBoost

### Model Comparison

| Model               | Best F1 Score | Best Threshold |
| ------------------- | ------------: | -------------: |
| Logistic Regression |         0.331 |            0.6 |
| Random Forest       |         0.356 |            0.3 |
| **XGBoost**         |     **0.537** |        **0.4** |

XGBoost achieved the best fraud-class F1 score and was selected as the final model.

## 🏆 Final Model Performance

The final model is:

**XGBoost with classification threshold = 0.4**

### Fraud Class Performance

| Metric    |   Score |
| --------- | ------: |
| Precision | **75%** |
| Recall    | **42%** |
| F1 Score  | **54%** |

### Overall Performance

| Metric      |   Score |
| ----------- | ------: |
| Accuracy    | **97%** |
| Macro F1    | **76%** |
| Weighted F1 | **97%** |

### Why F1 Score?

Because fraud detection is an imbalanced classification problem, accuracy alone can be misleading.

Therefore, this project focuses primarily on:

* Precision
* Recall
* F1 Score

rather than relying only on accuracy.

## 🎚️ Threshold Optimization

The default classification threshold of 0.5 was tested along with several alternative thresholds.

| Threshold | Precision |    Recall |        F1 |
| --------: | --------: | --------: | --------: |
|       0.1 |     0.180 |     0.684 |     0.285 |
|       0.2 |     0.326 |     0.557 |     0.411 |
|       0.3 |     0.537 |     0.456 |     0.493 |
|   **0.4** | **0.750** | **0.418** | **0.537** |
|       0.5 |     0.812 |     0.329 |     0.468 |
|       0.6 |     0.857 |     0.228 |     0.360 |
|       0.7 |     1.000 |     0.127 |     0.225 |

A threshold of **0.4** produced the highest F1 score.

The application therefore classifies a transaction as fraud when:

```text
Fraud Probability >= 0.40
```

Otherwise:

```text
Fraud Probability < 0.40
→ Legitimate
```

## 🧠 Features Used

The final model uses transaction, account, merchant, location, device, and transaction-type information.

### Numerical Features

* `amount`
* `hour`
* `account_age_days`
* `transactions_last_24h`
* `avg_transaction_amount`
* `failed_transactions`

### Transaction Features

* `is_international`
* `card_present`
* Transaction type encoded features

### Merchant Features

* Healthcare
* Grocery
* Travel
* Fuel
* Entertainment
* Electronics
* Clothing
* Restaurant

### Location Features

* Chennai
* Mumbai
* Bengaluru
* Kolkata
* Hyderabad
* Delhi
* Vijayawada
* Pune

### Device Features

* Mobile
* Desktop
* Tablet
* POS Terminal

## 💻 Streamlit Application

The project includes an interactive Streamlit application where users can enter transaction details such as:

* Transaction amount
* Average transaction amount
* Transaction hour
* International transaction
* Account age
* Recent transaction count
* Failed transactions
* Card presence
* Transaction type
* Merchant category
* Location
* Device type

The application returns:

* Fraud probability
* Fraud / legitimate classification
* Transaction details

### Example

```text
Fraud Probability: 10.61%

✅ LEGITIMATE TRANSACTION
```

If the predicted probability is greater than or equal to 40%:

```text
🚨 FRAUDULENT TRANSACTION DETECTED
```

## 📁 Project Structure

```text
credit_fraud_app/
│
├── app.py
├── requirements.txt
├── README.md
│
└── model/
    ├── xgb_model.pkl
    ├── scaler.pkl
    └── threshold.pkl
```

## 📦 Installation

Clone the repository:

```bash
git clone <your-github-repository-url>
```

Navigate to the project directory:

```bash
cd credit_fraud_app
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## 📋 Requirements

The main dependencies are:

```text
streamlit
pandas
numpy
scikit-learn
xgboost
joblib
```

## 🔮 Future Improvements

Potential improvements for future versions include:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Probability calibration
* More advanced fraud-specific feature engineering
* Cost-sensitive learning
* Explainable AI using SHAP
* Real-time transaction monitoring
* Model drift detection
* Cloud deployment
* Database integration
* Fraud alert and notification system

## 💼 Business Impact

A fraud detection system can help financial organizations:

* Identify suspicious transactions
* Reduce potential financial losses
* Improve transaction monitoring
* Prioritize suspicious transactions for manual investigation
* Improve fraud prevention workflows

## 👨‍💻 Author

**Bunny**

Aspiring Data Scientist focused on Machine Learning, Data Science, and AI.

## ⭐ Project Highlights

* End-to-end machine learning project
* Imbalanced classification problem
* SMOTE-based class balancing
* Multiple model comparison
* Threshold optimization
* XGBoost final model
* Real-time Streamlit prediction
* Deployment-ready model pipeline
