_# 💓 Heart Disease Prediction System

**Advanced web-based AI application for real-time heart disease prediction**.
Built using **Python, FastAPI, CNN (TensorFlow/Keras)**, and a **modern machine learning pipeline**.

---

## 🚀 Project Overview

This project presents a robust, web-based application designed for **predicting heart disease** using user-provided health parameters. The system integrates:

* **FastAPI microservice** architecture for high-performance API endpoints.
* **Pre-trained CNN model** for accurate classification.
* **Feature engineering, scaling, and dimensionality reduction** for optimal prediction performance.
* **User-friendly interface** for real-time prediction and results visualization.

It demonstrates a **practical AI application in healthcare diagnostics**.

---

## 🏗️ Architectural Components

The application is structured into **five main components**:

1. **User Interface Layer**

   * Accepts user input through a web form (`index.html`) with **11 health parameters**.
   * Displays prediction results on `result.html`.
2. **API Layer**

   * Built with **FastAPI**.
   * Routes:

     * `GET /` → Render input form.
     * `POST /predict` → Run ML pipeline and return prediction.
3. **Machine Learning Pipeline**

   * **Data Ingestion & Feature Engineering** → Converts form input into a Pandas DataFrame and generates **18 interaction features**.
   * **Feature Scaling** → Normalizes data with `StandardScaler`.
   * **Dimensionality Reduction** → Reduces features using `PCA`.
   * **CNN Classification** → Pre-trained CNN outputs probability of heart disease.
4. **Model & Preprocessor Storage**

   * Stores `scaler.pkl`, `pca.pkl`, and `heart_cnn_model.h5`.
   * Loaded at startup for **efficient inference**.
5. **Deployment & Infrastructure**

   * Containerized with **Docker**.
   * Deployable on cloud platforms (AWS, Azure, GCP).
   * Supports **load balancing, CI/CD**, monitoring, and logging.

---

## 📊 Data Flow & Pipeline

**Diagram**:
<img width="1663" height="748" alt="image" src="https://github.com/user-attachments/assets/98d81a39-c141-4c37-bed7-3a0b2b2d0bce" />


## 🧠 Machine Learning Pipeline Details

### 1. Data Ingestion & Feature Engineering

* Converts raw form data into **Pandas DataFrame**.
* Creates **interaction features** (e.g., `age × resting_bp`) to improve model learning.

### 2. Feature Scaling

* Applies `StandardScaler` to **normalize feature values**, ensuring balanced input to the CNN.

### 3. Dimensionality Reduction

* Uses `PCA` to reduce high-dimensional features into **principal components**, retaining maximum variance.

### 4. CNN Classification

* Reshapes PCA-transformed features into **3D array** for CNN input.
* Outputs **probability of heart disease**, converted to a **binary prediction**.

---

## 💾 Model & Preprocessor Storage

| Artifact             | Purpose                  |
| -------------------- | ------------------------ |
| `scaler.pkl`         | Feature scaling          |
| `pca.pkl`            | Dimensionality reduction |
| `heart_cnn_model.h5` | Trained CNN classifier   |

* All models are **pre-loaded** in FastAPI for **fast inference**.

---

## ⚙️ Deployment & Infrastructure

1. **Containerization**

   * Docker ensures consistent dev/test/prod environments.

2. **Cloud Deployment**

   * Deployable on AWS, Azure, or GCP.
   * Supports horizontal scaling for high traffic.

3. **Load Balancing**

   * Distributes requests across multiple instances.

4. **CI/CD Pipeline**

   * Automated builds, testing, and deployment using GitHub Actions or Jenkins.

5. **Monitoring & Logging**

   * Integration with **Prometheus/Grafana** and **ELK Stack** for performance tracking.

6. **Data Governance & Security**

   * HIPAA/GDPR compliant.
   * HTTPS, access control, and anonymization of sensitive patient data.

7. **MLOps**

   * Model versioning, experiment tracking, and automated retraining.

---

## 📂 File Structure

```
heart_disease_prediction/
│
├── backend/
│   ├── app.py                # FastAPI app
│   ├── models/
│   │   ├── scaler.pkl
│   │   ├── pca.pkl
│   │   └── heart_cnn_model.h5
│   └── requirements.txt
│
├── frontend/
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   └── static/
│       └── css, js, images
│
├── docker/
│   └── Dockerfile
│
└── README.md
```

---

## 💡 Key Features

* Real-time heart disease prediction.
* Scalable FastAPI microservice architecture.
* Advanced feature engineering for improved accuracy.
* CNN-based classification.
* Cloud-ready, Dockerized deployment.
* CI/CD, logging, and monitoring included.

---

Published Paper: “Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach Enhanced by Feature Engineering”
Published in CMC – Computers, Materials & Continua (Q2, SCI, Scopus)
https://www.techscience.com/cmc/v82n3/59942


Heart disease includes a multiplicity of medical conditions that affect the structure, blood vessels, and general operation of the heart. Numerous researchers have made progress in correcting and predicting early heart disease, but more remains to be accomplished. The diagnostic accuracy of many current studies is inadequate due to the attempt to predict patients with heart disease using traditional approaches. By using data fusion from several regions of the country, we intend to increase the accuracy of heart disease prediction. A statistical approach that promotes insights triggered by feature interactions to reveal the intricate pattern in the data, which cannot be adequately captured by a single feature. We processed the data using techniques including feature scaling, outlier detection and replacement, null and missing value imputation, and more to improve the data quality. Furthermore, the proposed feature engineering method uses the correlation test for numerical features and the chi-square test for categorical features to interact with the feature. To reduce the dimensionality, we subsequently used PCA with 95% variation. To identify patients with heart disease, hyperparameter-based machine learning algorithms like RF, XGBoost, Gradient Boosting, LightGBM, CatBoost, SVM, and MLP are utilized, along with ensemble models. The model’s overall prediction performance ranges from 88% to 92%. In order to attain cutting-edge results, we then used a 1D CNN model, which significantly enhanced the prediction with an accuracy score of 96.36%, precision of 96.45%, recall of 96.36%, specificity score of 99.51% and F1 score of 96.34%. The RF model produces the best results among all the classifiers in the evaluation matrix without feature interaction, with accuracy of 90.21%, precision of 90.40%, recall of 90.86%, specificity of 90.91%, and F1 score of 90.63%. Our proposed 1D CNN model is 7% superior to the one without feature engineering when compared to the suggested approach. This illustrates how interaction-focused feature analysis can produce precise and useful insights for heart disease diagnosis.

