
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

### **✅ Week 4: Supervised Learning (Linear & Logistic Regression)**

**Status:** Completed

**Objective:** Transition from data engineering to algorithmic prediction by building, training, and evaluating fundamental continuous and categorical models.

### 🏗️ Task 1: Continuous Financial Forecasting (Linear Regression)
* **Dataset:** Ames Housing Dataset
* **Target Variable:** `SalePrice` (Continuous)
* **Engineering Highlights:**
  * Applied **IQR Capping** to `GrLivArea` and `LotArea` to prevent spatial outliers from skewing the gradient.
  * Resolved **Heteroscedasticity** by applying a logarithmic transformation (`np.log1p`) to the target variable to stabilize variance.
  * Engineered a mathematically pure feature matrix using `StandardScaler` and `OneHotEncoder`.
* **Evaluation:** Achieved a baseline **RMSE of $31,199.14**, proving high accuracy in mid-market predictions while exposing linear limitations in non-linear luxury markets.

### 🚢 Task 2: Behavioral Classification (Logistic Regression)
* **Dataset:** Titanic Survival Dataset
* **Target Variable:** `Survived` (Binary Classification)
* **Engineering Highlights:**
  * Designed a strict **Zero-Leakage Pipeline**, applying `KNNImputer` for missing `Age` values *only* after the Train/Test split.
  * Capped extreme ticket prices (`Fare`) using IQR methods to protect gradient calculations.
  * Encoded categorical behavioral data (`Sex`, `Embarked`) to establish strict numerical weights.
* **Evaluation:** Achieved an **Accuracy of 81.01%**. Analyzed the Confusion Matrix to identify algorithmic pessimism (higher False Negatives), justifying the upcoming pivot to non-linear tree-based models.

### ✅ Week 5: Advanced Machine Learning & Ensembles

**Status:** Completed
**Focus:** Advancing from baseline models to Decision Trees, Random Forests, and Gradient Boosting (XGBoost).

### 🏗️ Phase 1: Zero-Leakage Preprocessing (Tree-Optimized)
* **Engineering Highlights:**
  * Initialized the environment and loaded the datasets.
  * Executed Zero-Leakage Data Cleaning by dropping `PassengerId`, `Name`, `Ticket`, and `Cabin`.
  * Capped `Fare` IQR, mapped `Sex`, and One-Hot Encoded `Embarked`.
  * Applied production transformations including Mode Imputation for `Embarked` and Post-Split KNN Imputation for `Age`.
  * Bypassed feature scaling since tree-based algorithms do not require it.

### 🌲 Phase 2: The Non-Linear Foundation (Decision Trees)
* **Engineering Highlights:** Instantiated and trained an unconstrained Decision Tree Classifier. The model grew to a Depth of 25 with 153 Leaves to perfectly separate the training data.
* **Evaluation:** Train Accuracy (98.31%) vs. Test Accuracy (81.56%).
* **Analysis:** The performance drop of 16.75% provided absolute proof of High Variance (Overfitting). The model memorized the training noise instead of the real survival boundaries.

### 🌳 Phase 3: Ensemble Learning (Bagging / Random Forest)
* **Engineering Highlights:** Instantiated and trained a Random Forest Classifier consisting of 100 estimators.
* **Evaluation:** Train Accuracy (98.31%) vs. Test Accuracy (81.01%).
* **Analysis:** The performance drop of 17.31% exposed the "Overfitting Committee". Because the 100 trees were unconstrained, they all overfitted individually, which amplified the variance rather than curing it.

### 🚀 Phase 4: Ensemble Learning (Boosting / Gradient Boosting)
* **Engineering Highlights:** Instantiated and trained a sequential Gradient Boosting Classifier with a default constraint of `max_depth=3`.
* **Evaluation:** Train Accuracy (90.45%) vs. Test Accuracy (81.01%).
* **Analysis:** The performance drop was successfully slashed to 9.44%, curing the variance. However, the model became too cautious (yielding only 45 True Positives) and struggled to find edge-case survivors.

### 🎛️ Phase 5: Hyperparameter Tuning (The Ultimate Masterpiece)
* **Engineering Highlights:**
  * Configured `GridSearchCV` to mathematically search for optimal Random Forest constraints, finding: `max_depth = 3`, `min_samples_split = 10`, `n_estimators = 100`.
  * Executed Grid Search on Gradient Boosting, finding optimal parameters: `learning_rate = 0.1`, `max_depth = 4`, `n_estimators = 100`.
* **Evaluation:**
  * **Tuned Random Forest:** Train (84.27%) vs. Test (79.33%). Variance was perfectly stabilized (4.94% drop), but the model became too simple (High Bias).
  * **Tuned Gradient Boosting:** Train (94.10%) vs. Test (82.12%). 
* **Analysis:** Achieved the Ultimate Tradeoff with Gradient Boosting. By increasing the complexity to `max_depth=4`, a slight amount of variance was reintroduced (11.98% gap), but it secured the highest Test Accuracy of the week (82.12%).

### 🏆 Phase 6: Model Comparison & Final Deployment Selection
* **Winning Model:** Tuned Gradient Boosting.
* **Justification:** It achieved the highest absolute testing accuracy (82.12%) while expertly balancing Precision (0.82) and Recall (0.68). It successfully navigated the Bias-Variance tradeoff through rigorous hyperparameter tuning.
### ✅ Week 6: Model Tuning & Validation

**Status:** Completed
**Focus:** Hyperparameter tuning via GridSearch, Stratified K-Fold Cross-Validation, Advanced OOP Scikit-Learn Pipelines, and balancing the Bias-Variance tradeoff for medical diagnostics.
**Dataset:** Pima Indians Diabetes Dataset (Binary Classification)

### 🏗️ Phase 1: Advanced Pipeline Architecture & Defensive Programming
* **Engineering Highlights:**
  * Engineered custom Stateful Transformers (e.g., `IQRCapperTransformer`) to calculate statistical boundaries strictly on training data, actively preventing data leakage into the testing set.
  * Implemented **Defensive Programming** by forcing explicit column declarations via conditional checks (`ValueError`) to prevent silent failures in production environments.
  * Solved the classic "NumPy Array Trap" by configuring Scikit-Learn to retain Pandas DataFrames (`transform_output="pandas"`, `verbose_feature_names_out=False`), ensuring feature names survived the `ColumnTransformer` destruction.
  * Assembled a Master `Pipeline` seamlessly linking `ZeroToNanTransformer`, Imputation routing, IQR Capping, and the final Estimator.

### 📈 Phase 2: Visual EDA & Non-Linear Feature Engineering
* **Engineering Highlights:**
  * Executed Zero-Leakage Pairplots strictly on the training set to visually identify non-linear biological boundaries.
  * Discovered critical interaction effects and complex clustering between `Glucose`, `BMI`, and `Age`.
  * Injected `PolynomialFeatures(degree=2)` into the pipeline for these specific markers. This mathematically bent the linear boundary, immediately boosting the baseline Logistic Regression Recall from 57.95% to 59.36%.

### 🩺 Phase 3: Baseline Cross-Validation & Variance Detection
* **Engineering Highlights:**
  * Deployed `StratifiedKFold(n_splits=5)` integrated with `cross_validate` to expose the true split variance of the models across different realities.
  * **Analysis:** The unconstrained baseline Decision Tree exhibited a massive **+/- 17.28% variance in Recall**, proving it was wildly overfitting to training noise across different folds. 

### 🎛️ Phase 4: Hyperparameter Tuning (The Medical Logic)
* **Engineering Highlights:**
  * Transitioned to an Ensemble **Random Forest Classifier** to cure the Decision Tree's high variance.
  * Implemented algorithmic penalties using `class_weight='balanced'` to force the mathematical gradient to heavily punish missed diabetic diagnoses.
  * Configured `GridSearchCV` to test hundreds of hyperparameter combinations across 5 folds, explicitly overriding the default accuracy metric to optimize strictly for `refit='recall'`.

### 🏆 Phase 5: Final Evaluation & Business Value Translation
* **Winning Model:** Tuned Random Forest Classifier
* **Evaluation Metrics:** Test Accuracy (73.38%), **Recall (78%)**, Precision (59%).
* **Analysis:** Deliberately traded overall accuracy and precision to maximize Recall. Out of 54 actual diabetic patients, the optimized model successfully caught 42 (True Positives), missing only 12. This architectural decision prioritizes human life (minimizing fatal missed diagnoses) over the minor inconvenience of secondary blood tests (False Positives).

### 💼 Phase 6: Professional Development & Portfolio Integration
* **Milestone:** Published Phase 3 Edge-AI diagnostic pipeline insights via the LinkedIn "Build-in-Public" framework. 
* **Documentation:** Rendered a highly customized Seaborn Heatmap of the Confusion Matrix to visually communicate the Precision-Recall tradeoff to non-technical stakeholders and global recruiters. 

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

