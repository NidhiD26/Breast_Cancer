from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global mlp_model, scaler, evaluation_metrics, test_samples
    try:
        data = pd.read_csv('Breast_cancer_data.csv')
        input_data = data[feature_names]
        target = data['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(input_data, target, test_size=0.3, random_state=42)
        
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        test_samples = {
            "features": X_test_reset.to_dict(orient='records'),
            "labels": y_test_reset.to_dict()
        }

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp_model = MLPClassifier(hidden_layer_sizes=(7,), activation="relu", max_iter=2000, random_state=42)
        mlp_model.fit(X_train_scaled, y_train)

        predictions = mlp_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions, output_dict=True)

        evaluation_metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "dataset_info": {
                "total_samples": len(data),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "feature_names": feature_names,
                "class_distribution": target.value_counts().to_dict()
            },
            "model_structure": get_model_structure(mlp_model)
        }
        print("Model and scaler trained and loaded successfully!")

    except Exception as e:
        print(f"Error during startup: {e}")
        mlp_model = None
        scaler = None
    yield

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
origins = [
    "http://localhost",
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    "http://127.0.0.1"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global Variables ---
mlp_model: MLPClassifier = None
scaler: StandardScaler = None
evaluation_metrics = {}
test_samples = {}
feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float

# --- Helper Functions ---
def get_model_structure(mlp: MLPClassifier):
    if not mlp: return []
    # Correctly get the number of neurons in each layer
    layer_sizes = [mlp.n_features_in_]
    layer_sizes.extend(mlp.hidden_layer_sizes)
    layer_sizes.append(mlp.n_outputs_)
    return layer_sizes

def relu(x):
    return np.maximum(0, x)

# --- FastAPI Events ---
@app.on_event("startup")
async def startup_event():
    global mlp_model, scaler, evaluation_metrics, test_samples
    try:
        data = pd.read_csv('Breast_cancer_data.csv')
        input_data = data[feature_names]
        target = data['diagnosis']

        X_train, X_test, y_train, y_test = train_test_split(input_data, target, test_size=0.3, random_state=42)
        
        X_test_reset = X_test.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        test_samples = {
            "features": X_test_reset.to_dict(orient='records'),
            "labels": y_test_reset.to_dict()
        }

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        mlp_model = MLPClassifier(hidden_layer_sizes=(7,), activation="relu", max_iter=2000, random_state=42)
        mlp_model.fit(X_train_scaled, y_train)

        predictions = mlp_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions, output_dict=True)

        evaluation_metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "dataset_info": {
                "total_samples": len(data),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "feature_names": feature_names,
                "class_distribution": target.value_counts().to_dict()
            },
            "model_structure": get_model_structure(mlp_model)
        }
        print("Model and scaler trained and loaded successfully!")

    except Exception as e:
        print(f"Error during startup: {e}")
        mlp_model = None
        scaler = None

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "Breast Cancer Prediction API"}

@app.get("/analysis")
async def get_analysis():
    if not evaluation_metrics:
        raise HTTPException(status_code=503, detail="Analysis data not available.")
    return evaluation_metrics

@app.get("/samples")
async def get_samples(num_samples: int = 5):
    if not test_samples:
        raise HTTPException(status_code=503, detail="Sample data not available.")
    
    num_available = len(test_samples["features"])
    if num_samples > num_available:
        num_samples = num_available
    
    random_indices = np.random.choice(num_available, num_samples, replace=False)
    
    samples = []
    for i in random_indices:
        samples.append({
            "features": test_samples["features"][i],
            "actual_label": "Malignant" if test_samples["labels"][i] == 1 else "Benign"
        })
        
    return samples

@app.post("/predict")
async def predict(input_data: PredictionInput):
    if mlp_model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or scaler not loaded.")

    try:
        features = np.array(list(input_data.dict().values())).reshape(1, -1)
        scaled_features = scaler.transform(features)
        
        # Manual forward pass to get hidden layer activations
        hidden_layer_input = np.dot(scaled_features, mlp_model.coefs_[0]) + mlp_model.intercepts_[0]
        hidden_layer_activations = relu(hidden_layer_input).flatten().tolist()

        # Get prediction from the model
        prediction = mlp_model.predict(scaled_features)[0]
        prediction_proba = mlp_model.predict_proba(scaled_features)[0].tolist()
        result = "Benign" if prediction == 0 else "Malignant"

        return {
            "prediction": int(prediction),
            "prediction_label": result,
            "prediction_probabilities": {
                "Benign": prediction_proba[0],
                "Malignant": prediction_proba[1] if len(prediction_proba) > 1 else 0
            },
            "features_used": dict(input_data),
            "hidden_layer_activations": hidden_layer_activations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")