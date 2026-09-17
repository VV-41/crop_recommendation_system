# 🌱 Crop AI & Agricultural Recommendation Assistant

> An intelligent, full-stack machine learning web application designed to help farmers and agronomists make data-driven crop selection decisions using granular soil chemistry, real-time meteorological insights, and an interactive command-based chatbot.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Random%20Forest-orange)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 🚀 Overview

Selecting the right crop based on environmental and soil conditions is critical for maximizing agricultural yield and sustainability. **Crop AI** bridges the gap between complex agronomic data and actionable advice. It combines a secure user authentication portal, a live weather dashboard, an interactive command assistant, and a high-performance **Random Forest Classifier** wrapped inside a robust Scikit-Learn `Pipeline`.

---

## ✨ Key Features

* **🔐 Secure User Authentication:** Dedicated login and registration system to keep user predictions and history private.
* **🤖 Advanced ML Pipeline:** Powered by a tuned **Random Forest Classifier** with categorical handling (`OneHotEncoder` with rare category grouping) and class-weight balancing.
* **💬 Command-Based Chatbot:** An integrated assistant supporting quick commands (e.g., `/help`, `/tips [crop]`, `/ask NPK`, `/recommend`) for rapid guidance.
* **🌦️ Real-Time Weather Integration:** Live tracking of environmental parameters like temperature and wind speed directly on the dashboard.
* **📊 Multi-Variate Inputs:** Processes 11 critical features including Nitrogen, Phosphorus, Potassium, Rainfall, Temperature, Humidity, Soil pH, Soil Type, Season, State, and Previous Crop history.


## 🛠️ Machine Learning Architecture

The modeling core (`model.py`) executes a standardized transformation and prediction workflow:

```text
[ Raw CSV Data ] 
       │
       ▼
[ ColumnTransformer ]
   ├── Numerical Features (Phosphorus, Potassium, Rainfall, Temp, Humidity, pH, Nitrogen)
   │     ├── SimpleImputer (strategy="median")
   │     └── StandardScaler
   │
   └── Categorical Features (Previous_Crop, State, Season, Soil_Type)
         ├── SimpleImputer (strategy="most_frequent")
         └── OneHotEncoder (handle_unknown="ignore", min_frequency=0.01)
       │
       ▼
[ RandomForestClassifier ]
   ├── n_estimators = 500 (tunable via RandomizedSearchCV)
   ├── class_weight = "balanced" (handles class imbalances)
   └── Regularization via min_samples_leaf & min_samples_split
