
# **AnalystLab Africa \- Machine Learning Engineering Portfolio**

## **📌 Overview**

This repository serves as my living, week-by-week portfolio for the AnalystLab Africa Machine Learning Internship Program. It documents my journey from raw data preprocessing and exploratory data analysis (EDA) through advanced hyperparameter tuning, and ultimately, live model deployment via APIs.

The core engineering philosophy of this repository focuses on building mathematically pure, memory-optimized pipelines suitable for scalable Machine Learning and resource-constrained Edge/TinyML environments.

## **🗂️ Directory Architecture**

Designed following the industry-standard Cookiecutter Data Science framework:

* data/raw/: Immutable, original datasets (Ignored via .gitignore).  
* data/processed/: Cleaned, ML-ready numerical matrices (Ignored via .gitignore).  
* notebooks/: Jupyter Notebooks containing the mathematical and algorithmic pipelines.  
* reports/: Executive summary PDFs detailing engineering decisions and data insights.  
* src/: (Upcoming) Source code for deployed model APIs (Flask/FastAPI).

## **🚀 Internship Roadmap & Weekly Logs**

### ✅ Weeks 1 & 2: Data Preprocessing & EDA

**Status:** Completed
**Datasets:** Titanic (Binary Classification) & Iris (Multi-Class Classification)
**Engineering Highlights:**

* **Missing Data Imputation:** Handled multivariate missingness using Median (numerical resistance to skew) and Mode (categorical) imputation. Dropped Titanic's `Cabin` column due to 77% data sparsity.

* **Outlier Boundary Mathematics:** Capped severe distribution skew in Titanic's `Fare` feature using the IQR (Interquartile Range) boundary method, while intentionally retaining biological variance outliers in Iris's `SepalWidthCm`.

* **Categorical Encoding:** Translated string labels to numerical states via `LabelEncoder` (Iris target) and solved the Dummy Variable Trap using One-Hot Encoding with `drop_first=True` (Titanic `Embarked`).

* **Feature Optimization:** Identified severe multicollinearity using Pearson Correlation Heatmaps. Flagged $r = -0.72$ between `Pclass` and `Fare`, and $r = 0.96$ between `PetalLength` and `PetalWidth`, taking steps to reduce model footprint by eliminating redundant variables.

### ✅ Week 3: Machine Learning Fundamentals & Validation Logic

**Status:** Completed
**Focus:** Establishing strict evaluation protocols, fixing data leakage vulnerabilities, and proving the "Accuracy Paradox."
**Engineering Highlights:**

* **Data Leakage Resolution:** Refactored the preprocessing pipeline to execute a Stratified 80/20 `train_test_split` *before* scaling and imputation. Ensuring `StandardScaler` and `LabelEncoder` are fit strictly on training data prevents mathematical leakage from the test set.

* **Advanced Imputation:** Upgraded from static Median imputation to `KNNImputer` (K=5) on the Titanic dataset, eliminating artificial variance spikes by algorithmically predicting missing ages based on neighboring features.

* **The Accuracy Paradox:** Deployed intentional "Dummy" and "Flawed" predictors to prove that raw Accuracy is a deceptive metric on imbalanced datasets. Generated Confusion Matrices and Classification Reports to establish Precision and Recall as the true indicators of model health.

### **🔜 Week 4: Supervised Learning (Linear & Logistic Regression)**

**Focus:** Building baseline predictive models. Predicting numerical values (House Prices via RMSE) and categorical classes (Titanic Survival via Accuracy).

### **🔜 Week 5: Advanced Machine Learning & Ensembles**

**Focus:** Advancing from baseline models to Decision Trees, Random Forests, and Gradient Boosting (XGBoost).

### **🔜 Week 6: Model Tuning & Validation**

**Focus:** Hyperparameter tuning via Grid/Random Search, K-Fold Cross-Validation, and balancing the Bias-Variance tradeoff on the Pima Indians Diabetes dataset.

### **🔜 Week 7: API Model Deployment**

**Focus:** Serializing trained models (.pkl / .joblib) and exposing them to the real world via Flask or FastAPI endpoints.

### **🔜 Week 8: Capstone Project**

**Focus:** An end-to-end Machine Learning solution deployed via Streamlit/Gradio solving a real-world industry problem.

## **🛠️ Tech Stack**

* **Languages:** Python 3.x  
* **Data Engineering:** Pandas, NumPy  
* **Machine Learning:** Scikit-Learn, XGBoost  
* **Visualizations:** Seaborn, Matplotlib  
* **Deployment (Upcoming):** Flask, FastAPI, Streamlit

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFoAAAAaCAYAAAA38EtuAAADeElEQVR4Xu2YTUhUURTHZ7Cg6NPKhpyPOzMODK4qBoqgwEVFghFIi8JNBBGtQiWiICiibaAElUQSEWFIBFEUuQhqERWBC1u0qgiihS0E20TZ7/juxTt33nPGSFG7f/jz7j1f793zzrv3zMRiHh4eHh4ec4RCobA6k8mcVEr1wXNwk2vjoqGhYSU+/dh2JJPJlPjYzGaza42tlkncPuQXUqlUwY71XyCXy7F+NUzSjpGEZYxbuX5Avs21taGT9wlORLBX7CQO42fpdHoX482MH2l9N+q4E3bRYgkLvgEHZWyEzC+R+CdU3nLLtgzoS9i9U8FXYPMhHJEXKP7YPWB+FJc68aP61/MiXyMblxhO2MUJFppnwV+5nrblyNqrJQLdfhJ23JaVSqWlyK/BPTJXU1U/JtVs7JifVUFVd015R0D2NQxbm5qa0jKXKzdok+Cu7XwFz7ub5/3tJlqSqBPRYcttoGumYpO2TBKP/FRMbwmSeGQ9xHtq50XuJ/Hd+1ZADgICXJeNHYcvsBenqzrAe5hzfeYjTELdBUfJpwP2W+H9fD6/xtU5kO1qEP4ify2usgwY7JO3pzf3MR6oH9YzHoKjsNn1McDvMLafayWx3s7WKS2JDEtoZoaJ1pU7AA+5OhfYbCf2OOwTP1dfBoxP6CTLXvaTB9qJOM61TV6CjF2f2YQ8MM+TUE6LFUa2uI0xfTAx7wpL6EwTje0O+FFNU2ACqXYVFOPtRCKxwtVHAodeONLY2LjB1c0leLlbVOXpH0pse2BW/KISGiWPQBzbK/AN9vWu0sAclNhdnq6bqUCxWFyF0wt4JzaDCs4GvWpFpUVRKrXqJ/aXkC9RBV9kaKJhuy0PgxQZdiPwuZxdrl5gkszaz8SmvqZm1rbXMa2EGMLRrNPiVIP0l9z0YK0k/gEqYJ0b519AugYVfPKTPy4MZE2yNlmjkckLlx7YttPyyXNKOb24Ban4bmJ2ytgI9T2qvkjTa5r9eaFCknARvjLdglQf83vwbkwnjmQWmX9TQYdV1lFldItI4m7Zco048iPof4hvpvyQ/15T7ghwHuPhsLe8kKAPqMesZ4BrK7wJX8r/F8ZG/5cxDIfc9k37TIQlWgXbX9TPdPmhlHd9KiB7bdSetABRx6JLeruSX4OT+2gt0K1diyTV1Xl4eHh4eHh4eHh4zD3+AFIUGzB3Q2OVAAAAAElFTkSuQmCC>

[image2]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEoAAAAaCAYAAAAQXsqGAAADeUlEQVR4Xu1XSWhUQRD9gwqKGy5xyCy//yw4iAcjg4ggAUMUAxkNxoPbXRRRiKCQkxIkIAiSQ4ISCB7Ei+IpIigoelAwl4BRMAcXAjnpIehBheh7091JT/Nn5hvMcugHj+6uruqlpqp+j+c5ODg4OPwTUqnUKiHECfA2eD2TyRRsnWrI5/PrfN+/oGy7ObZ1FGJBEOyGTh/Yjz2aKbOVliyy2ex6HPwJ2NPQ0LAGl2lC/x3YaevagM42pdsFNsL2NMdYc6upF4/HV8OZQ5h7SBv0t6MdQbvf1FvSwGEv49Bv0G7QMoxPgu/xq8dNXRN0KnSew+4xI1KJl0N2H7zHvpIxkm6CL/ijUKAc+4d7K52lDTqHTsIl7pjydDq9C/LvaA+ZchOwbYXOtG2rHP+VkaPGRfR/MNq0TjKZTEF2DXPZWcsQMI+h2JbL5dIcs4VRO2SNtu58gpfhpUIuW74cL2PKTUCnxKgIsaWjGC0ljjF/BePfGO9VUcgUXWnahELVgVvgVRhNgH1YZEBtwHzP2DbzBePXti8bKjdRz1FgFx2CdphrQd7Le1MOjoNnvVrFHMoHGYbI/x1QnsICQ75MgafCCNkwwO44dL9EJdYaQf3I2+to1LhsXUdhPiPkD33Xm72wrlHl+qPrGMfgIOeV7T5wqlZq87JnlJOOCBWSEMfQttOJ7Ns28wXs3zZXR3nyzJegNw69gIJAfv6/hThqGuNWbYhxI/gZHA7qpaGQ74mxRCKx2Z5bKFRzSDV5CJZB75yK3k9gP9ijHFXis0DITGHqFbWR4Siyel0uFAprofBSVIZtXdD7apNI5Oe9WCyusNfRwOGz0Ju0HaIdBXab8igQMgBmSgjaQa41J0dxES4WGJ/MKMDFBTY8GpVY/zBq1EZ7HQ0jNSpSwJef/l9stYxvINSUhFdZj26AD/T7yFhv5h2FdY9h/NOXJaYMw1G1U09U1qdFBc5wypepk1GimJDp81o7AO+eTRiP8sLgHsoMp7Cgl23RdoKT4E61Fp8+WzB+6xuPSxGlmBOBfFuM8gD23EKDqYlLDOA8z3CuDuWkMfSbtI5yyiPIPjCqtRyyi+ArIV/yTLmPYIueN/RawAns0wueZ5+2Xr2yw3Dj5rZ8ERHj/zOmKxzRXKuuWeDfE6Kjnh0LO3QOUFc/tB0cHBwcHBwcHP4//gLNMyay8RwrCwAAAABJRU5ErkJggg==>