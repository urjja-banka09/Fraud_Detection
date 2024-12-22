# Fraud_Detection
## Overview
This project demonstrates a machine learning pipeline for detecting fraudulent transactions using the XGBoost classifier. It incorporates essential stages of model development, evaluation, and explainability, offering insights into the behavior of the model and its decision-making process. 
## Key Features  

### Data Preprocessing  
1. Analyzed the dataset and focused on:  
   - Distribution of numerical features.  
   - Analysis of categorical features.  
   - Detecting outliers.  
   - Target distribution.  
2. Normalized numerical features and encoded categorical variables.  

### Model Training  
1. Started with Logistic Regression to establish a baseline.  
2. Tuned hyperparameters for XGBoost using grid search.  
3. Balanced the dataset to handle class imbalance effectively.  

### Evaluation  
1. Evaluated model performance using metrics like **F1-score**, **Precision**, **Recall**, and **AUC-ROC**.  
2. Provided a comprehensive **confusion matrix** for error analysis.  

### Explainability  
1. Used XGBoost's **feature importance** to highlight the most impactful features.  
2. Leveraged **SHAP** to explain predictions at both global and local levels.  

### Reproducibility  
1. Modularized the code to facilitate easy experimentation.  
2. Included detailed instructions to set up the environment and reproduce results.  

## Dataset

The dataset was retrieved from an open-source website, Kaggle.com.Dataset which has the critical features for a fraudulent transaction.
<br>
<br>
<b>Dataset: </b>
<a href="https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/data">kaggle Dataset</a>

<br>
<br>

## Algorithm 
1. Logistic Regression (L.R.)
3. XGBoost(xgb)
4. Isolation Forest(unsupervided)


<br>
<br>

## Setup  

To reproduce the results and run the fraud detection pipeline, follow these steps:  

### 1. **Download the Files**  
- **Jupyter Notebook**:  
  Download the notebook file (`Main.ipynb`) and place it in your working repository.  

- **Dataset**:  
  Download the CSV dataset (`card_transdata.csv`) and save it in the same repository as the notebook.  

---

### 2. **Run the Jupyter Notebook**  
Launch Jupyter Notebook in the repository and open the downloaded notebook:  

1. Navigate to the `Main.ipynb` file in your browser.  
2. Ensure the dataset (`card_transdata.csv`) is in the same directory as the notebook.  
3. Run all the cells in sequence.  

---

### Outputs  
- The notebook generates outputs such as metrics, confusion matrices, and visualizations.  
- All outputs are displayed inline within the notebook.  

---

### Notes  
- Ensure your dataset file is named correctly (`card_transdata.csv`).  
- If the dataset structure differs from expectations, update preprocessing steps in the notebook.

## Key Choices and Trade-offs  

### 1. **Model Choice**  
- **XGBoost** was selected due to its strong performance in handling tabular data and its ability to manage class imbalance effectively.  

### 2. **Class Balancing**  
- To address the issue of imbalanced data, a combination of **oversampling** techniques and **hyperparameter tuning** was employed.  

### 3. **Explainability**  
- The pipeline incorporated **feature importance** to identify key drivers of fraud detection.  
- **SHAP (SHapley Additive exPlanations)** was utilized to provide both high-level overviews and detailed, instance-specific insights into model predictions.  



