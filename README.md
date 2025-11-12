# 🏦 Predicting Term Deposit Subscription Using Direct Marketing Campaign Data  
*A Machine Learning Approach to Optimize Bank Marketing Strategies*  

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-Neural--Network-red?logo=pytorch&logoColor=white)  
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Modeling-orange?logo=scikit-learn&logoColor=white)  
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)  

---

## 🚀 Project Summary
This project predicts whether a **bank customer will subscribe to a term deposit** following a direct marketing campaign.  
Using a dataset from a **Portuguese bank’s phone-based marketing efforts**, we analyze how demographics, account information, and campaign interaction features influence subscription decisions.

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
Direct marketing calls are expensive—**who should the bank contact?**  
We predict the probability that a client subscribes (`y ∈ {yes, no}`) using:
- Client attributes (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`)
- Campaign details (`duration`, `campaign`, `pdays`, `previous`, `poutcome`)
- Contact context (`contact`, `month`)

---

## 📚 Dataset
- **Source:** UCI Machine Learning Repository — Bank Marketing  
- **File Used:** `bank-full.csv` (`;` separated)  
- **Records:** 45,211  
- **Target:** `y` (term deposit subscription: `yes` / `no`)

| Category | Example Features |
|-----------|------------------|
| Client Demographics | `age`, `job`, `marital`, `education`, `default`, `housing`, `loan` |
| Campaign Details | `contact`, `month`, `duration`, `campaign`, `pdays`, `previous`, `poutcome` |
| Target | `y` |

---

## 🔍 Exploratory Data Analysis
- **Class imbalance:** ~**11.7% yes** vs **88.3% no**  
- **Numerical insights:**  
  - `balance` and `duration` are right-skewed with outliers.  
  - Longer call `duration` strongly correlates with higher conversion rates.  
- **Categorical insights:**  
  - Higher success among **single**, **educated**, and **non-defaulting** clients.  
  - **Cellular** contact type and **previous successful campaigns** yield better results.

---

## ⚙️ Feature Engineering
- **One-Hot Encoding:** `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- **Binary Flags:**  
  - `long_campaign_contact = (campaign > 5)`  
  - `recent_contact = (pdays < 10)`
- **Interactions:**  
  - `duration_campaign_interaction = duration × campaign`  
  - `contact_poutcome_interaction = contact + '_' + poutcome`
- **Age Binning:** `Youth`, `Adult`, `Middle-Aged`, `Senior`
- **Outliers:** handled using IQR capping for `balance` and `duration`.

---

## 🧮 Model Development
**Pipeline:**
- **Split:** 85% train/validation, 15% hidden test (stratified)
- **Imbalance:** handled with **SMOTE**
- **Scaling:** applied `StandardScaler` to numeric features

**Models Implemented**
| Model | Implementation | Notes |
|--------|----------------|-------|
| Logistic Regression | Custom (NumPy) | Gradient Descent + L2 Regularization |
| Naive Bayes | Custom (NumPy) | Bernoulli NB with Laplace Smoothing |
| Neural Network | PyTorch | MLP (1 hidden layer, 16 units, ReLU, BCEWithLogitsLoss, Adam optimizer) |

---

## 📊 Model Performance
| Metric | Logistic Regression | Naive Bayes | Neural Network |
|---------|---------------------|--------------|----------------|
| **Training Accuracy (%)** | 85.31 | 78.72 | **91.35** |
| **Validation Accuracy (%)** | 83.58 | 79.65 | **86.44** |
| **Hidden Test Accuracy (%)** | 82.82 | 79.32 | **86.08** |
| **ROC-AUC (Validation)** | 0.90 | 0.83 | **0.91** |
| **Precision (Class 1)** | 40.01 | 31.58 | **45.07** |
| **Recall (Class 1)** | **80.87** | 63.40 | 72.75 |
| **F1-Score (Class 1)** | 82.23 | 82.33 | **87.75** |

The notebook includes **ROC/PR curves**, **confusion matrices**, and **learning curves** for all models.

---

## 💡 Key Insights
- 📞 **Call duration** and **previous campaign outcome** are top predictors.  
- ⚖️ **SMOTE** improved minority recall but reduced precision slightly.  
- 🧠 **Neural Network** achieved the highest F1 and AUC but with slight overfitting.  
- ✅ **Logistic Regression** provides strong interpretability and balanced generalization.  

---

## 🗂️ Project Structure
```plaintext
.
├── term_deposit_prediction.ipynb   # End-to-end notebook (EDA → FE → modeling → evaluation)
└── README.md                       # This file
