import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

from preprocess import load_data, split_data, scale_features


# load and preprocess data
X, y = load_data()
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
X_train, X_val, X_test = scale_features(X_train, X_val, X_test)

# train model (same as training script)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

model.fit(X_train, y_train)

# predictions
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:,1]

# confusion matrix
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

plt.title("Confusion Matrix")
plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# show some misclassified examples
misclassified = np.where(preds != y_test)[0][:5]

print("\nExample misclassified customers:")
for i in misclassified:
    print(f"Predicted={preds[i]}  Actual={y_test.iloc[i]}")