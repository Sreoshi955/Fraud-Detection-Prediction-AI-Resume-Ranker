Fraud Detection Prediction Web App

A web-based Fraud Detection System built using XGBoost and deployed via Flask API to identify fraudulent credit card transactions in real time.

Overview

This project is designed to demonstrate how machine learning models can be effectively integrated into real-world applications. The web app uses a trained XGBoost classifier to detect fraudulent transactions based on the popular Credit Card Fraud Detection Dataset.

Features

* Detects fraudulent transactions with high accuracy
* Real-time predictions via a web form
* User-friendly interface built with HTML/CSS and Flask
* Easily deployable for integration into dashboards or production systems
* Serialized model using pickle for fast inference

Project Structure

fraud-detection-webapp/
│
├── model/
│   └── xgboost_model.pkl        # Serialized ML model
│
├── templates/
│   └── index.html               # Web form interface
│
├── app.py                       # Flask application
├── fraud_detection.ipynb        # Jupyter Notebook for model training
└── README.md                    # Project documentation


Model Performance

* Model: XGBoost Classifier
* Evaluation Metrics: Accuracy, Precision, Recall, F1-Score
* Achieved high performance on imbalanced data using techniques like:

  * SMOTE (optional)
  * Under/Over sampling
  * Threshold tuning

Future Improvements

* Integration with a database to log predictions
* RESTful API endpoints for batch predictions
* Deployment on cloud (Render/AWS/GCP)
* Dashboard integration with Streamlit or Dash
