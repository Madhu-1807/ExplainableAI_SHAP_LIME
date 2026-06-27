# ExplainableAI: SHAP & LIME Explanations for ML Models

A comprehensive machine learning explainability project that demonstrates how to interpret model predictions using **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** on a real-world breast cancer response prediction dataset.

## 📋 Table of Contents
- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Models Trained](#models-trained)
- [Explainability Methods](#explainability-methods)
- [Installation](#installation)
- [Usage](#usage)
- [Results & Insights](#results--insights)
- [Key Findings](#key-findings)

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting pathologic complete response (PCR) in breast cancer patients, with a strong focus on model interpretability. The project compares five different machine learning models and explains their predictions using both SHAP and LIME methods, enabling clinicians and researchers to understand how models make decisions.

## 📁 Project Structure

```
ExplainableAI_SHAP_LIME/
├── XAI_SHAP_LIME_Project.ipynb  # Main notebook with complete pipeline
├── README.md                      # This file
└── outputs/
    ├── confusion_matrices.png     # Model performance visualization
    ├── shap_*.png                 # SHAP explanation plots
    └── lime_*.png                 # LIME explanation plots
```

## 📊 Dataset

- **Name:** ISPY (Investigation of Serial Studies to Predict Your Therapeutic Response with Imaging and Molecular Analysis) Data Paper
- **Size:** Clinical features with PCR (Pathologic Complete Response) labels
- **Target:** Binary classification (PCR=1: Responder, PCR=0: Non-Responder)
- **Preprocessing:**
  - Missing values dropped for target variable
  - Subject IDs removed (not features)
  - Remaining NaNs filled with column mean
  - Features standardized using StandardScaler
  - 80/20 train-test split with stratification (handles class imbalance)

## 🔧 Methodology

### Step 1: Data Preprocessing
- Load and clean the ISPY dataset from Excel
- Handle missing values and remove non-feature columns
- Separate features (X) and target (y)
- Apply StandardScaler normalization
- Split data with stratification to maintain class balance

### Step 2: Model Training
Train five different machine learning models with class balancing:
1. **Logistic Regression** - Baseline linear model
2. **Support Vector Machine (SVM)** - Non-linear classifier with RBF kernel
3. **Random Forest** - Ensemble method with 100 trees
4. **Decision Tree** - Single tree classifier
5. **XGBoost** - Gradient boosting classifier

All models use `class_weight='balanced'` to handle class imbalance.

### Step 3: Model Evaluation
- Evaluate using Accuracy, Precision, Recall, and F1-Score
- Generate confusion matrices for visual performance assessment
- Stratified cross-validation to ensure robust results

### Step 4: SHAP Explanations
Generate global and local SHAP explanations for each model:
- **Logistic Regression:** LinearExplainer with interventional perturbation
- **SVM:** KernelExplainer on 50 background samples + 20 test samples
- **Random Forest & Decision Tree:** TreeExplainer with tree path-dependent perturbation
- **XGBoost:** TreeExplainer optimized for gradient boosting

Outputs include:
- Bar plots showing top 15 feature importance
- Summary plots showing feature contribution distribution

### Step 5: LIME Explanations
Generate local model-agnostic explanations:
- Create instance-specific explanations for individual predictions
- Analyze one PCR=1 (responder) and one PCR=0 (non-responder) sample
- Generate interpretable local linear explanations
- Top 10 most important features visualized per instance

## 🤖 Models Trained

| Model | Description | Key Advantage |
|-------|-------------|--------------|
| **Logistic Regression** | Linear baseline | Fast, interpretable coefficients |
| **SVM (RBF)** | Non-linear classifier | Effective on high-dimensional data |
| **Random Forest** | Ensemble of trees | Captures feature interactions |
| **Decision Tree** | Single tree | Most interpretable structure |
| **XGBoost** | Gradient boosting | State-of-the-art performance |

## 📖 Explainability Methods

### SHAP (SHapley Additive exPlanations)
SHAP values provide a unified measure of feature importance based on cooperative game theory:
- **Global Explanations:** Feature importance across all predictions
- **Local Explanations:** Individual feature contributions per prediction
- **Summary Plots:** Show both magnitude and direction of feature effects

**Visualizations:**
- Bar plots: Average absolute SHAP value impact
- Summary plots: Feature contribution distribution and value correlation

### LIME (Local Interpretable Model-agnostic Explanations)
LIME explains individual predictions by fitting local linear approximations:
- Model-agnostic: Works with any model
- Local focus: Explains individual predictions
- Interpretable: Uses actual feature values

**Visualizations:**
- Instance-specific feature contribution plots
- Shows which features drove specific predictions up or down

## 🚀 Installation

### Prerequisites
- Python 3.7+
- Jupyter Notebook or Google Colab

### Required Libraries
```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost shap lime
```

Or install from requirements:
```bash
pip install -r requirements.txt
```

## 📝 Usage

### Run the Complete Pipeline
```python
# Open the notebook in Google Colab or Jupyter
# Run all cells in sequence:

1. Load and preprocess data
2. Train all 5 models
3. Visualize confusion matrices
4. Generate SHAP explanations
5. Generate LIME explanations
```

### Extract Specific Explanations
```python
# SHAP feature importance
shap.summary_plot(shap_values, X_test_df, plot_type="bar")

# LIME instance explanation
exp = explainer.explain_instance(sample, model.predict_proba, num_features=10)
exp.show_in_notebook()
```

## 📊 Results & Insights

### Model Performance Summary
- All models achieve competitive accuracy on the test set
- Class balancing improves recall for the minority class (PCR=1)
- Confusion matrices show false positive/negative trade-offs

### Feature Importance Rankings
SHAP analysis reveals the most influential features across different models:
- Top features identified consistently across models
- Feature interactions captured by tree-based models
- Linear vs. non-linear feature importance differences

### Individual Prediction Explanations
LIME plots show:
- How specific features influenced each prediction
- Feature value thresholds that drive classification
- Model-specific decision patterns

## 🔍 Key Findings

1. **Model Consistency:** SHAP feature importance rankings are largely consistent across models, validating robustness

2. **Class-Specific Features:** Different features drive PCR=1 vs PCR=0 predictions

3. **Feature Interactions:** Random Forest and XGBoost capture non-linear interactions better than linear models

4. **Explainability Trade-off:** Simpler models (Logistic Regression) are easier to interpret; complex models (XGBoost) often perform better but are harder to explain without tools like SHAP/LIME

5. **Clinical Relevance:** Key predictive features can be validated against medical literature and domain expertise

## 📚 References

- SHAP: [Lundberg & Lee (2017)](https://arxiv.org/abs/1705.07874) - "A Unified Approach to Interpreting Model Predictions"
- LIME: [Ribeiro et al. (2016)](https://arxiv.org/abs/1602.04938) - "Why Should I Trust You?"
- ISPY Dataset: Breast cancer response prediction study

## 👤 Author

Created by: Madhu-1807

## 📄 License

This project is open source and available for educational and research purposes.

## 💡 Future Enhancements

- [ ] SHAP interaction plots
- [ ] Decision plot visualizations
- [ ] Waterfall plots for individual predictions
- [ ] Feature interaction analysis
- [ ] Sensitivity analysis
- [ ] Model comparison reports
- [ ] Web-based dashboard for interactive exploration

---

**Note:** This project demonstrates explainability techniques on a medical dataset. Results should be validated with domain experts before clinical application.
