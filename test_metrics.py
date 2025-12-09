from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Scenario 1: Only Benign (0) samples
y_true = [0, 0, 0]
y_pred = [0, 0, 0]

print("--- Scenario 1: Only 0s ---")
try:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    print("Confusion Matrix (labels=[0, 1]):\n", cm)
except Exception as e:
    print("CM Error:", e)

try:
    cr = classification_report(y_true, y_pred, labels=[0, 1], output_dict=True, zero_division=0)
    print("Classification Report (labels=[0, 1]):\n", cr)
except Exception as e:
    print("CR Error:", e)

# Scenario 2: Mixed but missing one class in predictions
y_true_2 = [0, 1]
y_pred_2 = [0, 0] # Predicted Benign for Malignant sample

print("\n--- Scenario 2: Missing Pred Class ---")
cm2 = confusion_matrix(y_true_2, y_pred_2, labels=[0, 1])
print("Confusion Matrix:\n", cm2)
cr2 = classification_report(y_true_2, y_pred_2, labels=[0, 1], output_dict=True, zero_division=0)
print("Classification Report:\n", cr2)
