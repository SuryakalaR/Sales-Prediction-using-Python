# 📊 Sales Prediction using Python

This project predicts product sales based on advertising expenditure using Machine Learning. A Linear Regression model is trained on the Advertising dataset to estimate sales from TV, Radio, and Newspaper advertisement budgets. The project also includes an interactive Streamlit dashboard for visualization and prediction.

---

## 📝 Project Overview

This project performs the following tasks:

1. **Data Loading**
   - Loads the `advertising.csv` dataset using Pandas.

2. **Data Exploration**
   - Displays dataset information.
   - Checks missing values and duplicate records.
   - Generates summary statistics.

3. **Data Visualization**
   - Scatter plots
   - Histogram
   - Box plot
   - Correlation heatmap
   - Pair plot
   - Sales trend analysis

4. **Data Preprocessing**
   - Selects advertising features.
   - Splits the dataset into training and testing sets.

5. **Model Training**
   - Trains a Linear Regression model using Scikit-learn.

6. **Prediction**
   - Predicts product sales based on advertising budgets entered by the user.

7. **Model Evaluation**
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score
   - Feature Importance
   - Actual vs Predicted comparison

8. **Interactive Dashboard**
   - Built using Streamlit.
   - Allows users to enter advertisement budgets.
   - Displays predicted sales instantly.
   - Provides interactive charts and model performance metrics.

---

# 💾 Dataset

**Dataset Name:** Advertising Dataset

**File:** `advertising.csv`

### Dataset Description

The dataset contains advertising budgets spent on different media platforms and corresponding product sales.

### Dataset Features

| Feature | Description |
|----------|-------------|
| TV | Amount spent on TV advertisements |
| Radio | Amount spent on Radio advertisements |
| Newspaper | Amount spent on Newspaper advertisements |
| Sales | Product sales (Target Variable) |

---

# 🎯 Features and Target

### Input Features (X)

- TV
- Radio
- Newspaper

### Target Variable (Y)

- Sales

---

# ⚙️ Technologies Used

- Python 3.x
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- Seaborn
- Scikit-learn

---

# 📚 Machine Learning Algorithm

**Linear Regression**

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. In this project, it predicts product sales based on advertising expenditures.

---

# 📦 Required Libraries

Install the required libraries using:

```bash
pip install -r requirements.txt
```

### requirements.txt

```txt
streamlit
pandas
numpy
matplotlib
seaborn
plotly
scikit-learn
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/SuryakalaR/Sales-Prediction-using-Python.git
```

Move into project folder

```bash
cd Sales-Prediction-using-Python
```

Install libraries

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Run the Streamlit application

```bash
streamlit run sales_prediction.py
```

---

# 📊 Dashboard Modules

The application consists of four major modules.

## 🏠 Home

- Project Overview
- Dataset Information
- Dashboard Introduction

## 📈 Visualization

- Scatter Plot
- Histogram
- Box Plot
- Heatmap
- Pair Plot
- Sales Trend

## 🤖 Model

Displays

- Linear Regression Model
- Accuracy
- RMSE
- MSE
- R² Score
- Feature Importance
- Residual Plot
- Actual vs Predicted Plot

## 🔮 Prediction

Users can enter

- TV Advertisement Budget
- Radio Advertisement Budget
- Newspaper Advertisement Budget

The application predicts

- Expected Product Sales
- Performance Gauge
- Feature Contribution Chart

---

# 📊 Model Evaluation Metrics

### Mean Squared Error (MSE)

Measures the average squared prediction error.

### Root Mean Squared Error (RMSE)

Shows the average prediction error in the original units.

### R² Score

Measures how well the model explains the variation in sales.

Higher R² values indicate better model performance.

---

# 📈 Output

The dashboard displays

- Dataset Preview
- Data Statistics
- Interactive Charts
- Correlation Matrix
- Model Performance
- Predicted Sales
- Feature Importance
- Residual Analysis

---

# 💡 Future Enhancements

- Multiple Machine Learning Algorithms
- Random Forest Regression
- Decision Tree Regression
- XGBoost Regression
- Model Comparison
- Hyperparameter Tuning
- Cloud Deployment
- Real-Time Data Integration
- User Authentication
- Database Support

---

# 📂 Project Structure

```
Sales-Prediction-using-Python/
│
├── advertising.csv
├── sales_prediction.py
├── requirements.txt
├── README.md
└── screenshots/
```

