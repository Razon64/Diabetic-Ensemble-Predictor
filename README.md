# Diabetic-Ensemble-Predictor
A high-performance machine learning pipeline for diabetes risk assessment featuring Stacking Ensembles (TabNet, GRU, CatBoost) and SMOTE-driven class balancing. It utilizes a hybrid approach, combining deep learning architectures with gradient boosting to maximize prediction sensitivity in imbalanced medical datasets.

# Features
- Stacked Generalization: Combines TabNet, GRU, and CatBoost for diversified feature learning.
- Class Imbalance Handling: Implements SMOTE and custom class weighting (1:15) to improve recall for minority classes.
- Outlier Mitigation: Automated IQR-based filtering and median imputation.
- Explainable AI: Integrated SHAP/Feature Importance plotting to identify key medical risk factors.

# Models Used
1. TabNet: Deep learning for tabular data.
2. GRU (Gated Recurrent Unit): Neural network for sequential/numeric pattern recognition.
3. CatBoost: High-performance gradient boosting.
4. LightGBM: Optimized tree-based learning with SMOTE.


## Requirements
- pandas
- numpy
- torch
- scikit-learn
- catboost
- pytorch-tabnet
- imbalanced-learn (for SMOTE)
- lightgbm
