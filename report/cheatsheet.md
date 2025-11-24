# Project Cheatsheet: Breast Cancer Prediction Dashboard

## Key Project Features

-   **Full-Stack Application:** A complete system with a Python backend and a JavaScript frontend.
-   **Interactive Dashboard:** Provides a user-friendly interface to interact with the machine learning model.
-   **Real-time Prediction:** Users can input features and get immediate predictions from the ANN model.
-   **Model Analysis:** The dashboard displays key performance metrics (confusion matrix, classification report) and visualizes the ANN structure.
-   **Sample Data Loader:** Allows for easy testing of the prediction functionality with pre-loaded samples from the test set.
-   **RESTful API:** The backend is a well-structured API that separates the machine learning logic from the frontend presentation.

---

## Technology Stack

-   **Backend:** Python, FastAPI, Uvicorn, scikit-learn, Pandas, NumPy.
-   **Frontend:** HTML, CSS, JavaScript (ES6+), Chart.js.
-   **Core ML Model:** `MLPClassifier` (Artificial Neural Network).

---

## Possible Presentation Questions & Answers

#### Q1: Why did you choose FastAPI for the backend?

**A:** FastAPI is a modern, high-performance Python web framework. I chose it because it's very fast, easy to learn, and has automatic interactive documentation generation (like Swagger UI), which is great for developing and testing APIs. Its use of Pydantic for data validation also ensures that the data sent to the API is correct.

#### Q2: How does the model training work? Is it trained every time a user makes a prediction?

**A:** The model is trained only **once** when the backend application starts up. The training process involves loading the dataset, splitting it into training and testing sets, scaling the features, and then training the `MLPClassifier` model. This ensures that prediction requests are fast, as they only need to perform a forward pass through the already-trained model, not retrain it.

#### Q3: What are the limitations of the current model or project?

**A:**
-   **Static Dataset:** The model is trained on a static CSV file. It doesn't learn from new predictions.
-   **Simple Model:** The `MLPClassifier` is relatively simple. More complex models like Gradient Boosting or more advanced deep learning architectures might yield better performance.
-   **No Model Persistence:** The model is retrained every time the server restarts. In a production environment, a trained model should be saved to a file (e.g., using `joblib` or `pickle`) and loaded on startup to ensure consistency and save time.
-   **Basic Frontend:** The frontend is built with vanilla JS. A framework like React or Vue could make it more scalable and maintainable.

#### Q4: How could you improve the project?

**A:**
-   **Model Persistence:** Save the trained model and scaler to disk and load them on startup.
-   **CI/CD Pipeline:** Implement a CI/CD pipeline to automate testing and deployment.
-   **More Advanced Model:** Experiment with other models to see if performance can be improved.
-   **User Authentication:** Add user accounts to save prediction history.
-   **Database Integration:** Store the dataset and prediction results in a database.
-   **Frontend Framework:** Rebuild the frontend using a modern framework like React or Vue for better component management.

#### Q5: What is CORS and why was it needed?

**A:** CORS stands for Cross-Origin Resource Sharing. It's a security feature implemented by web browsers to prevent a web page from making requests to a different domain than the one that served the page. In this project, the frontend is served from `http://localhost:8001` and the backend from `http://localhost:8000`. Because these are different "origins", the browser would block the frontend's API requests to the backend unless the backend explicitly allows it. I added the `CORSMiddleware` in FastAPI to tell the browser that it's safe to accept requests from the frontend's origin.

#### Q6: How is the ANN visualized on the frontend?

**A:** The backend has an `/analysis` endpoint that returns the model's structure (e.g., `[5, 7, 2]` for 5 input neurons, 7 hidden neurons, and 2 output neurons). The frontend then uses the **Chart.js** library, but not for its charting capabilities. Instead, it uses the HTML5 `<canvas>` element to programmatically draw the neurons as circles and the connections as lines based on the structure data received from the backend. This is a custom drawing implementation in JavaScript.
