__1. Web Interface:__

- The user interface is a simple HTML form (`templates/index.html`) where a user can input the following medical information:

  - Age
  - Sex
  - Chest Pain Type
  - Resting Blood Pressure
  - Cholesterol
  - Fasting Blood Sugar
  - Resting ECG
  - Max Heart Rate Achieved
  - Exercise Induced Angina
  - Oldpeak (ST depression induced by exercise relative to rest)
  - ST Slope

- After submitting the form, the application displays the prediction result on a separate page (`templates/result.html`).

__2. Backend Logic (`app.py`):__

- The backend is a FastAPI application that defines two main endpoints:

  - `/`: This is the root endpoint that serves the main `index.html` page.
  - `/predict`: This endpoint receives the user's input from the form via a POST request.

- The prediction process involves the following steps:

  1. The application receives the input data and creates a pandas DataFrame.
  2. It then engineers a set of interaction features by multiplying different input variables.
  3. The data is scaled using a pre-trained scikit-learn `StandardScaler` (`models/scaler.pkl`).
  4. Principal Component Analysis (PCA) is applied for dimensionality reduction using a pre-trained PCA model (`models/pca.pkl`).
  5. The processed data is then fed into a pre-trained Keras CNN model (`models/heart_cnn_model.h5`) for prediction.
  6. The model outputs a prediction, which is then interpreted as either "Heart Disease Detected" or "No Heart Disease".

__3. Machine Learning Model:__

- The core of the application is a pre-trained Convolutional Neural Network (CNN) model.
- The model, along with the scaler and PCA transformer, is loaded from the `models/` directory when the application starts.

__4. Dependencies (`requirements.txt`):__

- The project relies on several key Python libraries, including:

  - `fastapi`: For building the web application.
  - `uvicorn`: As the ASGI server to run the FastAPI application.
  - `tensorflow` and `keras`: For loading and using the CNN model.
  - `scikit-learn`: For data preprocessing (scaling and PCA).
  - `pandas`: For data manipulation.
  - `numpy`: For numerical operations.

In summary, this project provides a user-friendly web interface for a powerful machine learning model that can predict the presence of heart disease. It demonstrates a full-stack approach, combining a web framework with a deep learning model to create a practical and useful application.


Published Paper: “Harmonization of Heart Disease Dataset for Accurate Diagnosis: A Machine Learning Approach Enhanced by Feature Engineering”
Published in CMC – Computers, Materials & Continua (Q2, SCI, Scopus)
https://www.techscience.com/cmc/v82n3/59942


Heart disease includes a multiplicity of medical conditions that affect the structure, blood vessels, and general operation of the heart. Numerous researchers have made progress in correcting and predicting early heart disease, but more remains to be accomplished. The diagnostic accuracy of many current studies is inadequate due to the attempt to predict patients with heart disease using traditional approaches. By using data fusion from several regions of the country, we intend to increase the accuracy of heart disease prediction. A statistical approach that promotes insights triggered by feature interactions to reveal the intricate pattern in the data, which cannot be adequately captured by a single feature. We processed the data using techniques including feature scaling, outlier detection and replacement, null and missing value imputation, and more to improve the data quality. Furthermore, the proposed feature engineering method uses the correlation test for numerical features and the chi-square test for categorical features to interact with the feature. To reduce the dimensionality, we subsequently used PCA with 95% variation. To identify patients with heart disease, hyperparameter-based machine learning algorithms like RF, XGBoost, Gradient Boosting, LightGBM, CatBoost, SVM, and MLP are utilized, along with ensemble models. The model’s overall prediction performance ranges from 88% to 92%. In order to attain cutting-edge results, we then used a 1D CNN model, which significantly enhanced the prediction with an accuracy score of 96.36%, precision of 96.45%, recall of 96.36%, specificity score of 99.51% and F1 score of 96.34%. The RF model produces the best results among all the classifiers in the evaluation matrix without feature interaction, with accuracy of 90.21%, precision of 90.40%, recall of 90.86%, specificity of 90.91%, and F1 score of 90.63%. Our proposed 1D CNN model is 7% superior to the one without feature engineering when compared to the suggested approach. This illustrates how interaction-focused feature analysis can produce precise and useful insights for heart disease diagnosis.


