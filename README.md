# PredictingTermDepositSubscription
🏦 Predicting Term Deposit Subscription Using Direct Marketing Campaign Data
A Machine Learning Approach to Optimize Bank Marketing Strategies




🚀 Project Summary
This project predicts whether a bank customer will subscribe to a term deposit following a direct marketing campaign.
Using a dataset from a Portuguese bank’s phone-based marketing efforts, we explored how customer demographics, financial attributes, and past interactions influence subscription decisions.
🎯 Goal: Build predictive models that help banks target customers more effectively, reduce marketing costs, and improve conversion rates.
📦 Table of Contents
🎯 Problem Statement
📚 Dataset
🔍 Exploratory Data Analysis
⚙️ Feature Engineering
🧮 Model Development
📊 Model Performance
💡 Key Insights
🛠️ Tools Used
🗂️ Project Structure
👩‍💻 Authors
📄 References
🎯 Problem Statement
Banks collect massive amounts of customer data but often struggle to use it effectively in marketing.
In this study, we focus on improving the targeting of term deposit offers by predicting whether a client will subscribe after being contacted.
The project combines data preprocessing, feature engineering, and supervised learning to enhance decision-making and optimize marketing strategies.
📚 Dataset
Source: UCI Machine Learning Repository – Bank Marketing Dataset
File Used: bank-full.csv
Records: 45,211 customer records
Features: 17 attributes (demographic, campaign, and economic variables)
Target Variable: y (binary — yes if the customer subscribed to a term deposit, no otherwise)
Category	Feature	Description
Client Demographics	age, job, marital, education, default, housing, loan	Customer personal & financial details
Campaign Details	contact, month, duration, campaign, pdays, previous, poutcome	Information from marketing interactions
Target	y	Subscription outcome (yes/no)
🔍 Exploratory Data Analysis
Key findings from EDA include:
Class Imbalance: Only 11.7% of customers subscribed (y='yes'), while 88.3% did not.
Numerical Insights:
age: Mean 40.9 years; distribution right-skewed.
balance: Range −8019 to 102,127; significant outliers.
duration: Longer calls correlated with higher likelihood of subscription.
Categorical Insights:
Higher subscription rates among single, educated, and non-defaulting customers.
Cellular contacts and previous campaign success increased conversion probability.
⚙️ Feature Engineering
One-Hot Encoding: Applied to categorical features (job, marital, education, etc.).
Binary Flags:
long_campaign_contact → if campaign > 5
recent_contact → if pdays < 10
Interaction Terms:
duration_campaign_interaction = duration × campaign
contact_poutcome_interaction = contact + poutcome
Binning:
age grouped into Youth, Adult, Middle-Aged, Senior.
Outliers in balance and duration were capped using the IQR method for improved stability.
🧮 Model Development
Data Preprocessing
Train/Validation Split: 85% train-validation, 15% hidden test (stratified by target).
SMOTE: Applied to balance the minority class.
Scaling: StandardScaler used for numerical features.
Models Implemented
Model	Type	Library	Notes
Logistic Regression	Custom Implementation	NumPy	Gradient Descent + L2 Regularization
Naive Bayes	Custom	NumPy	With Laplace Smoothing
Neural Network	Feedforward (1 hidden layer)	PyTorch	16 hidden neurons, ReLU activation
📊 Model Performance
Metric	Logistic Regression	Naive Bayes	Neural Network
Training Accuracy (%)	85.31	78.72	91.35
Validation Accuracy (%)	83.58	79.65	86.44
Hidden Test Accuracy (%)	82.82	79.32	86.08
ROC-AUC (Validation)	0.90	0.83	0.91
Precision (Class 1)	40.01	31.58	45.07
Recall (Class 1)	80.87	63.40	72.75
F1-Score (Class 1)	82.23	82.33	87.75
💡 Key Insights
📞 Call duration and previous campaign success are the most influential features.
📉 Imbalanced data was mitigated using SMOTE, improving recall but slightly reducing precision.
🧠 Neural Network outperformed other models in F1-score and AUC but showed slight overfitting.
⚖️ Logistic Regression offers strong generalization and interpretability—ideal for deployment.
📊 Naive Bayes achieved decent recall but was limited by its independence assumptions.
🛠️ Tools Used
Python 3.8+
Pandas, NumPy, Matplotlib, Seaborn
Scikit-Learn
Imbalanced-Learn (SMOTE)
PyTorch
SciPy
