# 🏦 Predicting Term Deposit Subscription Using Direct Marketing Campaign Data  
*A Machine Learning Approach to Optimize Bank Marketing Strategies*  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-Neural--Network-red?logo=pytorch&logoColor=white)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikit-learn&logoColor=white)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)  

---

## 🚀 Project Summary
This project predicts whether a **bank customer will subscribe to a term deposit** following a direct marketing campaign.  
Using a dataset from a **Portuguese bank’s phone-based marketing efforts**, we analyze how demographics, account info, and campaign interaction features influence subscription decisions.

> 🎯 **Goal:** Build predictive models that help banks target customers more effectively, reduce marketing costs, and improve conversion rates.

---

## 📦 Table of Contents
- [🎯 Problem Statement](#-problem-statement)
- [📚 Dataset](#-dataset)
- [🔍 Exploratory Data Analysis](#-exploratory-data-analysis)
- [⚙️ Feature Engineering](#️-feature-engineering)
- [🧮 Model Development](#-model-development)
- [📊 Model Performance](#-model-performance)
- [💡 Key Insights](#-key-insights)
- [🗂️ Project Structure](#️-project-structure)
- [👩‍💻 Author](#-author)
- [📄 References](#-references)

---

## 🎯 Problem Statement
Direct marketing calls are expensive; **who should the bank contact?**  
We predict the probability that a client subscribes (`y ∈ {yes, no}`) using:
- Client attributes (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`)
- Campaign attributes (`duration`, `campaign`, `pdays`, `previous`, `poutcome`)
- Contact context (`contact`, `month`)

---

## 📚 Dataset
- **Source:** [UCI Machine Learning Repository – Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
- **File used:** `bank-full.csv` (`;` separated)  
- **Records:** 45,211  
- **Target:** `y` (term deposit subscription: `yes` / `no`)

| Category | Example Features |
|---|---|
| Client Demographics | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Campaign Details | `contact`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome` |
| Target | `y` |

---

## 🔍 Exploratory Data Analysis
- **Class imbalance:** ~**11.7% yes** vs **88.3% no**  
- **Numerical:** `balance` and `duration` right-skewed with outliers; longer `duration` associates with higher conversion  
- **Categorical:** higher subscription among **single**, **higher education**, **no default**; **cellular** contact and **previous success** drive better outcomes

---

## ⚙️ Feature Engineering
- **One-Hot Encoding:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **Binary flags:**  
  - `long_campaign_contact = (campaign > 5)`  
  - `recent_contact = (pdays < 10)`
- **Interactions:**  
  - `duration_campaign_interaction = duration × campaign`  
  - `contact_poutcome_interaction = contact + '_' + poutcome`
- **Age binning:** `Youth`, `Adult`, `Middle-Aged`, `Senior`
- **Outliers:** IQR capping for `balance` and `duration`

---

## 🧮 Model Development
**Data protocol**
- **Split:** 85% train/validation, **15% hidden test** (stratified)
- **Imbalance:** **SMOTE** on training only
- **Scaling:** `StandardScaler` for numeric features

**Models**
- **Custom Logistic Regression** (NumPy; Gradient Descent + L2)
- **Custom Bernoulli Naive Bayes** (binarized inputs; Laplace smoothing)
- **PyTorch MLP** (1 hidden layer, 16 units, ReLU, BCEWithLogitsLoss, Adam)

---

## 📊 Model Performance
*(Validation + hidden test from the notebook; fill in if you re-run with different splits.)*

| Metric | Logistic Regression | Naive Bayes | Neural Network |
|---|---:|---:|---:|
| **Training Accuracy (%)** | 85.31 | 78.72 | **91.35** |
| **Validation Accuracy (%)** | 83.58 | 79.65 | **86.44** |
| **Hidden Test Accuracy (%)** | 82.82 | 79.32 | **86.08** |
| **ROC-AUC (Validation)** | 0.90 | 0.83 | **0.91** |
| **Precision (Class 1)** | 40.01 | 31.58 | **45.07** |
| **Recall (Class 1)** | **80.87** | 63.40 | 72.75 |
| **F1-Score (Class 1)** | 82.23 | 82.33 | **87.75** |

Also included in the notebook: **confusion matrices**, **ROC & PR curves**, and **learning curves**.

---

## 💡 Key Insights
- 📞 **Call duration** and **previous campaign success** are most influential  
- ⚖️ **SMOTE** improves recall of the minority class but can lower precision  
- 🧠 **Neural Network** achieves best overall metrics but shows mild overfitting  
- ✅ **Logistic Regression** remains strong, interpretable, and deployment-friendly

---

## 🗂️ Project Structure
```plaintext
.
├── term_deposit_prediction.ipynb   # End-to-end notebook (EDA → FE → models → evaluation)
└── README.md                       # This file
