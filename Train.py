import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier

# Load normalized features and labels
features, labels, label_encoder = joblib.load('audio_features_augmented.pkl')

# Load scaler (important for future inference)
scaler = joblib.load('scaler.pkl')

# (Optional) Check stats
print(f"Mean: {np.mean(features):.4f}, Std: {np.std(features):.4f}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# Train model
#model = KNeighborsClassifier(n_neighbors=5)
#model = SVC(kernel='rbf', probability=True, random_state=42)
#model = SVC(kernel='rbf', probability=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'trained_model.joblib')

# Predict and evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Accuracy: {accuracy:.2f}%")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.ax_.set_title('Confusion Matrix')
plt.show()
