# Aarya Shah - Machine Learning Projects 
## AIML303 - Machine Learning and Pattern Recognition

This repository contains a collection of my machine learning projects, demonstrating my skills in various techniques including supervised and unsupervised learning, data preprocessing, and model evaluation.

---

### **Project 1 - Fraud Detection on Credit Card Dataset**

* **Goal:** The primary objective of this project is to build a classification model that can accurately identify fraudulent credit card transactions from a given dataset. The model is designed to minimize both false negatives and false positives, which is crucial for real-world applications.
* **Methodology:** The project involves data analysis, which revealed a heavily imbalanced dataset. A classification model was built using a range of libraries, including `seaborn`, `matplotlib.pyplot`, `pandas`, `numpy`, `sklearn.model_selection` (for `train_test_split`), `sklearn.preprocessing` (`StandardScaler`), `sklearn.metrics`, `sklearn.linear_model` (`LogisticRegression`), `sklearn.tree` (`DecisionTreeClassifier`), `sklearn.ensemble` (`RandomForestClassifier`), and `sklearn.discriminant_analysis` (`LinearDiscriminantAnalysis`).

### **Project 2 - Food Delivery System using Clustering**

* **Goal:** This project focuses on analyzing customer preferences and behavior in food delivery services through exploratory data analysis and clustering techniques. The aim is to uncover insights that can be used for customer segmentation and understanding cuisine preferences.
* **Methodology:** The notebook uses several key libraries for data manipulation and machine learning, including `numpy`, `matplotlib.pyplot`, `pandas`, `seaborn`, `sklearn.decomposition` (`PCA`), `sklearn.preprocessing` (`StandardScaler`), and `sklearn.cluster` (`KMeans`, `AgglomerativeClustering`, `DBSCAN`).

### **Project 3 - Diabetes Detection using Classification**

* **Goal:** The project's goal is to develop and evaluate a K-Nearest Neighbors (KNN) classifier to detect diabetes. The process includes data preprocessing, addressing class imbalance, and optimizing the model.
* **Methodology:** The notebook uses `numpy`, `matplotlib.pyplot`, `pandas`, `seaborn`, `sklearn.preprocessing` (`StandardScaler`), `sklearn.metrics`, `sklearn.model_selection` (`train_test_split`), `sklearn.neighbors` (`KNeighborsClassifier`), and `imblearn.over_sampling` (`SMOTE`). A key finding was that several features like BMI, Skin Thickness, Glucose, Insulin, and Blood Pressure had zero values, which were identified as null values requiring handling.

### **Project 4 - Predicting Online Purchase Intent**

* **Goal:** The main goal of this project is to build a machine learning model to predict whether an e-commerce website visitor will make a purchase.
* **Methodology:** The approach is based on the KNN algorithm. The project involves data preprocessing, such as converting categorical columns like 'Month', 'VisitorType', 'Weekend', and 'Revenue' into numerical types using a `LabelEncoder`. The notebook also identifies a significant class imbalance in the 'Revenue' column, with a much larger number of sessions not resulting in a purchase. The libraries used include `pandas`, `numpy`, `seaborn`, `matplotlib.pyplot`, `sklearn.preprocessing` (`StandardScaler`, `LabelEncoder`), `sklearn.model_selection` (`train_test_split`, `cross_val_score`), `sklearn.decomposition` (`PCA`), `sklearn.linear_model` (`LogisticRegression`), `sklearn.ensemble` (`RandomForestClassifier`), `sklearn.tree` (`DecisionTreeClassifier`), `sklearn.neighbors` (`KNeighborsClassifier`), `imblearn.over_sampling` (`SMOTE`), and `sklearn.metrics` for evaluation.

### **Project 5 - Advertisement Budget Using Linear Regression**

* **Goal:** The goal of this project is to predict sales based on advertisement budgets using linear regression.
* **Methodology:** This project utilizes `numpy`, `matplotlib.pyplot`, `seaborn`, `pandas`, `sklearn.model_selection` (`train_test_split`), `sklearn.linear_model` (`LinearRegression`), `sklearn.metrics`), and `sklearn.preprocessing` (`StandardScaler`). The notebook performs exploratory data analysis, noting that the maximum sales amount is significantly larger than the mean, which could indicate a potential area for further analysis.
