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
    global mlp_model, scaler, evaluation_metrics, test_samples, accumulated_y_true, accumulated_y_pred
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

        mlp_model = MLPClassifier(hidden_layer_sizes=(7,), activation="relu", max_iter=2000, random_state=42, alpha=1.0)
        mlp_model.fit(X_train_scaled, y_train)

        predictions = mlp_model.predict(X_test_scaled)
        cm = confusion_matrix(y_test, predictions)
        cr = classification_report(y_test, predictions, output_dict=True)
        # print(cr)

        evaluation_metrics = {
            "confusion_matrix": cm.tolist(),
            "classification_report": cr,
            "dataset_info": {
                "total_samples": len(data),
                "training_samples": len(X_train),
                "testing_samples": len(X_test),
                "feature_names": feature_names,
                "class_distribution": {str(k): v for k, v in target.value_counts().to_dict().items()}
            },
            "model_structure": get_model_structure(mlp_model)
        }

        # Initialize cumulative lists with the test set predictions for ongoing evaluation
        accumulated_y_true = y_test.tolist()
        accumulated_y_pred = predictions.tolist()
        
        print(f"Initialized session tracking with {len(accumulated_y_true)} samples.")
        print("Model and scaler trained and loaded successfully!")

    except Exception as e:
        print(f"Error during startup: {e}")
        mlp_model = None
        scaler = None
    yield

app = FastAPI(lifespan=lifespan)

# --- CORS Middleware ---
origins = ["*"]
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
accumulated_y_true = []
accumulated_y_pred = []
feature_names = ['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area', 'mean_smoothness']

from typing import Optional
from pydantic import BaseModel, conlist

# --- Pydantic Models ---
class PredictionInput(BaseModel):
    mean_radius: float
    mean_texture: float
    mean_perimeter: float
    mean_area: float
    mean_smoothness: float
    actual_label: Optional[str] = None


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
        features_dict = input_data.dict()
        actual_label_str = features_dict.pop("actual_label", None)
        
        # --- Start Debug Logging ---
        print("\n--- New Prediction Request ---")
        print(f"Received features: {features_dict}")
        # --- End Debug Logging ---

        # Ensure the features are in the correct order
        ordered_features = [features_dict[name] for name in feature_names]
        features = np.array(ordered_features).reshape(1, -1)
        
        # --- Start Debug Logging ---
        print(f"Ordered features: {features}")
        # --- End Debug Logging ---
        
        scaled_features = scaler.transform(features)
        
        # --- Start Debug Logging ---
        print(f"Scaled features: {scaled_features}")
        # --- End Debug Logging ---
        
        hidden_layer_input = np.dot(scaled_features, mlp_model.coefs_[0]) + mlp_model.intercepts_[0]
        hidden_layer_activations = relu(hidden_layer_input).flatten().tolist()

        prediction = mlp_model.predict(scaled_features)[0]
        
        # --- Start Debug Logging ---
        print(f"Prediction: {prediction}")
        print("--- End Prediction Request ---\n")
        # --- End Debug Logging ---

        prediction_proba = mlp_model.predict_proba(scaled_features)[0].tolist()
        result = "Benign" if prediction == 0 else "Malignant"

        prediction_quality = None
        updated_metrics_response = None
        if actual_label_str:
            actual_label_val = 1 if actual_label_str == "Malignant" else 0
            if prediction == 1 and actual_label_val == 1:
                prediction_quality = "True Positive"
            elif prediction == 0 and actual_label_val == 0:
                prediction_quality = "True Negative"
            elif prediction == 1 and actual_label_val == 0:
                prediction_quality = "False Positive"
            elif prediction == 0 and actual_label_val == 1:
                prediction_quality = "False Negative"

            # --- Update Cumulative Metrics ---
            global accumulated_y_true, accumulated_y_pred, evaluation_metrics
            
            accumulated_y_true.append(actual_label_val)
            accumulated_y_pred.append(int(prediction))
            
            # Recalculate metrics
            # Recalculate metrics with fixed labels to handle sparse session data
            new_cm = confusion_matrix(accumulated_y_true, accumulated_y_pred, labels=[0, 1])
            new_cr = classification_report(accumulated_y_true, accumulated_y_pred, output_dict=True, labels=[0, 1], zero_division=0)
            
            # Update global metrics
            evaluation_metrics["confusion_matrix"] = new_cm.tolist()
            evaluation_metrics["classification_report"] = new_cr
            evaluation_metrics["dataset_info"]["total_samples"] += 1 # Optional: update total samples count
            if actual_label_str == "Benign":
                 evaluation_metrics["dataset_info"]["class_distribution"]['0'] += 1
            else:
                 evaluation_metrics["dataset_info"]["class_distribution"]['1'] += 1

            print("Metrics updated with new prediction.")
            updated_metrics_response = evaluation_metrics

        return {
            "prediction": int(prediction),
            "prediction_label": result,
            "prediction_probabilities": {
                "Benign": prediction_proba[0],
                "Malignant": prediction_proba[1] if len(prediction_proba) > 1 else 0
            },
            "features_used": features_dict,
            "hidden_layer_activations": hidden_layer_activations,
            "prediction_quality": prediction_quality,
            "updated_metrics": updated_metrics_response
        }
    except Exception as e:
        print(f"Prediction Error: {e}") # Log the error
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")