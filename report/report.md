# Project Report: Breast Cancer Prediction Dashboard

## 1. Project Overview

This project is a full-stack web application designed to predict breast cancer diagnosis based on a given set of features. It provides an interactive dashboard for users to:
-   Get predictions from a trained Artificial Neural Network (ANN) model.
-   View detailed analysis of the model's performance.
-   Understand the structure of the dataset and the ANN.
-   Load sample data to easily test the prediction functionality.

The application consists of a Python-based backend API that handles data processing and machine learning, and a vanilla HTML, CSS, and JavaScript frontend that provides the user interface.

---

## 2. Backend Architecture

The backend is a RESTful API built with Python and the FastAPI framework. It is responsible for training the model, serving predictions, and providing analytical data to the frontend.

### 2.1. Technology Stack

-   **Python 3:** Core programming language.
-   **FastAPI:** A modern, high-performance web framework for building APIs.
-   **Uvicorn:** An ASGI server to run the FastAPI application.
-   **scikit-learn:** Used for building and evaluating the machine learning model (`MLPClassifier`).
-   **Pandas:** For data manipulation and loading the dataset from a CSV file.
-   **NumPy:** For numerical operations, especially for handling feature arrays.

### 2.2. API Endpoints

The backend exposes the following endpoints:

-   **`GET /`**: A root endpoint that returns a welcome message.
-   **`POST /predict`**: The core prediction endpoint. It accepts a JSON object with 5 features and returns the model's prediction (Benign/Malignant) and prediction probabilities.
-   **`GET /analysis`**: Provides a detailed analysis of the model, including its structure, performance metrics (confusion matrix, classification report), and dataset information.
-   **`GET /samples`**: Returns a specified number of random samples from the test set, which the frontend uses to populate the "Load Sample" feature.

### 2.3. Model Training & Evaluation
 
-   **Model:** A Multi-layer Perceptron Classifier (`MLPClassifier`) from scikit-learn is used. The model has one hidden layer with 7 neurons and uses the ReLU activation function.
-   **Training Process:**
    1.  The `Breast_cancer_data.csv` dataset is loaded on application startup.
    2.  The data is split into a training set (70%) and a testing set (30%).
    3.  A `StandardScaler` is fitted on the training data to normalize the features.
    4.  The model is trained on the scaled training data.
    5.  The trained model is then evaluated on the scaled testing data, and the results (confusion matrix, classification report) are stored in memory.
-   **Data Preprocessing:** `StandardScaler` is used to standardize features by removing the mean and scaling to unit variance. This is crucial for neural networks as they are sensitive to feature scaling.

---

## 3. Frontend Architecture

The frontend is a single-page dashboard built with standard web technologies. It communicates with the backend API to fetch data and display it in an interactive and user-friendly manner.

### 3.1. Technology Stack

-   **HTML5:** Provides the structure of the dashboard.
-   **CSS3:** Used for styling the dashboard, including the layout, cards, and form elements. A CSS Grid layout is used for the main dashboard structure.
-   **JavaScript (ES6+):** Handles all the client-side logic, including API calls (`fetch`), DOM manipulation, and rendering of visualizations.
-   **Chart.js:** A simple yet flexible JavaScript charting library used to visualize the ANN structure.

### 3.2. Dashboard Components

-   **Prediction Card:** Contains a form for users to input the 5 required features. It also includes a sample loader to pre-fill the form with data from the test set.
-   **ANN Structure Card:** Displays a visual representation of the neural network's architecture (input layer, hidden layer, output layer).
-   **Model Performance Card:** Shows the confusion matrix and classification report, providing a clear view of the model's accuracy, precision, and recall.
-   **Dataset Information Card:** Provides an overview of the dataset, including the number of samples, features, and class distribution.

### 3.3. Interaction Flow

1.  **On Page Load:** The JavaScript makes API calls to `GET /analysis` and `GET /samples`.
2.  The data from `/analysis` is used to populate the "ANN Structure", "Model Performance", and "Dataset Information" cards.
3.  The data from `/samples` is used to populate the "Load Sample" dropdown.
4.  **User Interaction:**
    -   The user can select a sample and click "Load Sample" to fill the prediction form.
    -   The user can manually enter feature values and click "Predict".
    -   This triggers a `POST` request to the `/predict` endpoint.
    -   The prediction result is then displayed in the "Prediction Card".

---

## 4. How to Run the Project

### 4.1. Backend Setup

1.  Navigate to the `backend_api` directory.
2.  Create and activate a Python virtual environment.
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run the server: `uvicorn main:app --reload`

### 4.2. Frontend Setup

1.  Navigate to the `frontend` directory.
2.  Start a simple HTTP server: `python -m http.server 8001`
3.  Open a web browser and go to `http://localhost:8001`.
