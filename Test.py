import librosa
import numpy as np
import joblib

# --- Function: Spectral Entropy ---
def spectral_entropy(y, sr, n_short_blocks=10):
    ps = np.abs(np.fft.fft(y))**2
    ps_norm = ps / np.sum(ps)
    ps_split = np.array_split(ps_norm, n_short_blocks)
    entropy = -np.sum([np.sum(p * np.log2(p + 1e-10)) for p in ps_split])
    return entropy

# --- Function: Estimate Reverb Decay ---
def estimate_reverb_decay(y, sr):
    rms = librosa.feature.rms(y=y)[0]
    decay = np.polyfit(np.arange(len(rms)), 20 * np.log10(rms + 1e-10), deg=1)[0]
    return decay

# --- Load Model & Label Encoder ---
model = joblib.load('trained_model.joblib')
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')

# --- Load and Preprocess Test Audio ---
file_path = r'D:/AudioDetection/Dataset/Speakerphone/sp10.wav'
x, fs = librosa.load(file_path, sr=None)

# Normalize
x = x / np.max(np.abs(x))

# Resample to 16000 Hz if needed
if fs != 16000:
    x = librosa.resample(x, orig_sr=fs, target_sr=16000)
    fs = 16000

# --- Feature Extraction ---
mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13)
mfcc_mean = np.mean(mfcc, axis=1)  # shape (13,)

zcr = librosa.feature.zero_crossing_rate(x)
zcr_mean = np.mean(zcr)

centroid = librosa.feature.spectral_centroid(y=x, sr=fs)
centroid_mean = np.mean(centroid)

flatness = librosa.feature.spectral_flatness(y=x)
flatness_mean = np.mean(flatness)

entropy = spectral_entropy(x, fs)
reverb_decay = estimate_reverb_decay(x, fs)

# --- Final Feature Vector ---
features = np.concatenate([
    mfcc_mean,
    [zcr_mean, centroid_mean, flatness_mean, entropy, reverb_decay]
]).reshape(1, -1)  # shape: (1, 18)

print("Extracted feature vector shape:", features.shape)

# --- Predict Class ---
pred_label_encoded = model.predict(features)[0]
predicted_class = label_encoder.inverse_transform([pred_label_encoded])[0]


# --- Predict class with confidence scores ---
pred_proba = model.predict_proba(features)[0]  # Get probabilities
pred_label_encoded = np.argmax(pred_proba)     # Class with highest probability
predicted_class = label_encoder.inverse_transform([pred_label_encoded])[0]

# --- Display results ---
print(f"🟩 Predicted class: {predicted_class}")
print("📊 Confidence scores:")
for class_index, class_name in enumerate(label_encoder.classes_):
    print(f"   - {class_name}: {pred_proba[class_index]*100:.2f}%")

