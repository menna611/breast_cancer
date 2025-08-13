# Breast Cancer Detection

## Project Overview
This project focuses on predicting whether a tumor is **benign** or **malignant** based on patient data. Using machine learning models, the system classifies tumors to help with early detection and treatment decisions. The project demonstrates data cleaning, visualization, modeling, and evaluation of predictive performance.

## Dataset
The dataset used is the **Breast Cancer Wisconsin (Diagnostic) 

**Number of samples:** 569  
**Features:** 30 numeric features (e.g., radius, texture, smoothness, compactness, concavity, symmetry)  
**Target variable:** `diagnosis` (M = Malignant, B = Benign)

## Data Preprocessing
- Checked for missing values and handled them.  
- Converted categorical labels (`M`, `B`) to numeric using label encoding.  
- Standardized features to normalize scale differences.  
- Removed outliers to improve model performance.

## Exploratory Data Analysis (EDA)
- Visualized feature distributions using **histograms** and **boxplots**.  
- Analyzed correlations using a **heatmap** to identify important features.  
- Explored differences in feature statistics between benign and malignant tumors.

## Machine Learning Models
The following models were implemented:
  
1. **Random Forest Classifier** – ensemble model for better accuracy.  
2. **Decision Tree Classifier** – interpretable model for decision paths.  

**Performance Metrics:**
- Accuracy  
- Precision  
- Recall  
- F1-Score  
- Confusion Matrix  

