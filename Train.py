import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.svm import SVC
#from sklearn.tree import DecisionTreeClassifier
#from xgboost import XGBClassifier
import matplotlib.pyplot as plt

# Load features and labels
features, labels, label_encoder = joblib.load('audio_features_augmented.pkl')

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, stratify=labels, random_state=42
)

# Initialize and train model
#model = KNeighborsClassifier(n_neighbors=5)
#model = SVC(kernel='rbf', probability=True, random_state=42)
#model = SVC(kernel='rbf', probability=True, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'trained_model.joblib')

# Predict
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"✅ Accuracy: {accuracy:.2f}%")

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.ax_.set_title('Confusion Matrix')
plt.show()
