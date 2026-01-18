# Forecasting Individual Medical Insurance Charges

## 📌 Project Objective

The primary objective of this project is to **predict individual medical insurance charges** using demographic and lifestyle attributes. By applying machine learning regression techniques, this project aims to identify key cost drivers and build an accurate predictive model that can support risk assessment and pricing strategies in the healthcare insurance domain.

---

## 📖 Project Description

Healthcare costs vary significantly across individuals due to differences in age, lifestyle, and health conditions. This project analyzes a structured medical insurance dataset and applies an end-to-end data science workflow including data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation.

The project demonstrates practical experience in:
- Data analysis and visualization
- Supervised machine learning (regression)
- Model evaluation and comparison
- Writing clean, reproducible, and well-documented code

---

## 🧠 Business Problem

Insurance providers need to estimate future medical expenses accurately to:
- Set fair insurance premiums
- Manage financial risk
- Identify high-risk customer segments

This project addresses this problem by predicting insurance charges at an individual level using historical data.

---

## 📊 Dataset Information

**Dataset Name:** Medical Insurance Cost Personal Dataset

The dataset contains individual medical insurance records with the following attributes:

| Feature    | Description |
|------------|-------------|
| age        | Age of the insured individual |
| sex        | Gender of the individual |
| bmi        | Body Mass Index |
| children   | Number of dependents |
| smoker    | Smoking status (yes/no) |
| region     | Residential region |
| charges   | Annual medical insurance charges (target variable) |

**Target Variable:** `charges`  
**Problem Type:** Supervised Regression

---

## 📁 Repository Structure
---

## 🔍 Exploratory Data Analysis (EDA)

Key EDA steps include:
- Distribution analysis of numerical features
- Comparison of charges across smoker vs non-smoker groups
- Correlation analysis between features
- Identification of outliers and data skewness

Visualizations include histograms, box plots, scatter plots, and correlation heatmaps.

---

## ⚙️ Data Preprocessing

- Handling categorical variables using encoding techniques
- Feature scaling for numeric attributes
- Data consistency and validation checks
- Train-test data split

---

## 🤖 Modeling Approach

Multiple regression models were implemented and evaluated:

- Linear Regression
- Ridge and Lasso Regression
- Random Forest Regressor
- Gradient Boosting / XGBoost (if applicable)

Each model was trained using the same dataset split for fair comparison.

---

## 📈 Model Evaluation Metrics

Models were evaluated using standard regression metrics:

- **R² Score**
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**

Model performance comparison and insights are documented in the notebooks.

---

## 🧪 Results & Key Insights

- Smoking status is the strongest predictor of higher insurance charges
- BMI and age show a positive correlation with medical costs
- Tree-based models outperform linear models by capturing non-linear relationships

Detailed performance results and charts are available in the analysis notebooks.

---

## 🚀 How to Run the Project

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/mupparaju1985/Forecasting_Individual_Medical_Insurance_Charges.git
cd Forecasting_Individual_Medical_Insurance_Charges

### Step 2: Create Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

### Step 3: Install Dependencies
pip install -r requirements.txt

### Step 4: Run the Notebooks
jupyter notebook
