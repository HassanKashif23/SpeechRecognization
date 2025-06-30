import librosa
import numpy as np
import joblib

# --- Load Trained Model, Scaler, Label Encoder ---
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.pkl')  # 💡 NEW: load the same scaler used in training
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')

# --- Load and Preprocess Test Audio ---
file_path = r'D:/AudioDetection/App/nospeaker.wav'
x, sr = librosa.load(file_path, sr=None)

x = x / np.max(np.abs(x))  # Normalize

if sr != 16000:
    x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    sr = 16000

# --- Feature Extraction (19 features total) ---
mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)

zcr = np.mean(librosa.feature.zero_crossing_rate(x))
centroid = np.mean(librosa.feature.spectral_centroid(y=x, sr=sr))
flatness = np.mean(librosa.feature.spectral_flatness(y=x))
rolloff = np.mean(librosa.feature.spectral_rolloff(y=x, sr=sr))
rms = np.mean(librosa.feature.rms(y=x))

# Reverb Decay (slope of dB curve)
rms_energy = librosa.feature.rms(y=x)[0]
decay = np.polyfit(np.arange(len(rms_energy)), 20 * np.log10(rms_energy + 1e-10), deg=1)[0]

# Combine all features
feature_vector = np.hstack([
    mfcc_mean,
    zcr, centroid, flatness, rolloff, rms, decay
]).reshape(1, -1)

print("✅ Extracted feature vector shape:", feature_vector.shape)

# --- Apply Scaling (must be same as training) ---
scaled_features = scaler.transform(feature_vector)

# --- Predict Class ---
pred_proba = model.predict_proba(scaled_features)[0]
pred_label = np.argmax(pred_proba)
predicted_class = label_encoder.inverse_transform([pred_label])[0]

# --- Display ---
print(f"🟩 Predicted Class: {predicted_class}")
print("📊 Confidence Scores:")
for i, class_name in enumerate(label_encoder.classes_):
    print(f"   - {class_name}: {pred_proba[i]*100:.2f}%")
