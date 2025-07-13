# 🚗 Auto-MPG: Exploratory Data Analysis & Regression Dashboard

An interactive dashboard built with **Dash and Plotly** that focuses mainly on **exploratory data analysis (EDA)**, offering visualizations and intuitive model comparisons on the [Auto-MPG dataset](https://www.kaggle.com/datasets/uciml/autompg-dataset). It was aimed at understanding dynamic data exploration techniques, and visualization of lightweight regression models.

---

## 📁 Dataset

This project uses the data from the original **[Auto MPG dataset](https://archive.ics.uci.edu/dataset/9/auto+mpg)** from the UCI Machine Learning Repository. It includes attributes for various 1970s–80s vehicles, such as:

- **MPG** (Miles Per Gallon – target variable)
- **Horsepower**
- **Cylinders**
- **Displacement**
- **Acceleration**
- **Weight**
- **Model Year**
- **Origin**
- **Car Name** (used for reference only)

The dataset was preprocessed to handle missing values and encode categorical features (like `origin`) where necessary. 

---

## 📊 Key Features

- **Rich Visualizations**:
  - Scatter plots to analyze feature–target relationships
  - Box, histogram, and violin plots for distribution insights
  - Correlation heatmap to evaluate multicollinearity

- **Interactive Controls**:
  - Dropdowns and sliders to filter plots dynamically
  - Feature selection for regression models

- **Model Comparison**:
  - Regression models included: **Linear**, **Ridge**, **Lasso**, **ElasticNet**, **XGBoost**

---

## :camera: Sample Images

The following are some sample screenshots for EDA from the dashboard.

<p align="center">
  <img src="sample images/Dash-Scatter.png" alt="Scatterplot example" width="350">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="sample images/Dash-Violin.png" alt="Violin Plot example" width="350">
</p>

<p align="center">
  <em>Scatterplot</em>  
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;
  <em>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Violin Plot</em>
</p>

<br/>

## ⚙️ Tech Stack

- Python  
- Dash (Plotly)  
- Pandas, NumPy  
- Scikit-learn, XGBoost  
- Seaborn / Matplotlib (for EDA outside dashboard)

---

## 📌 Future Improvements

- Allow users to enter custom parameters 
- Add model tuning controls - improve flexibility
- Save user predictions/export reports 
---

Let me know if you'd like badges, hosted deployment links, or GIF previews added as well.


