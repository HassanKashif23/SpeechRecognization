import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm

# Dataset path and classes
dataset_path = r'D:\AudioDetection\augmented_dataset'
classes = ['Conversation', 'Speakerphone', 'Background']

features = []
labels = []

# --- Feature: Reverb Decay Function ---
def compute_reverb_decay(audio, sr):
    rms = librosa.feature.rms(y=audio)[0]
    peak_idx = np.argmax(rms)
    peak_energy = rms[peak_idx]
    decay_threshold = peak_energy * 0.1
    decay_time = 0.0
    for i in range(peak_idx + 1, len(rms)):
        if rms[i] < decay_threshold:
            decay_time = (i - peak_idx) * (512 / sr)
            break
    return decay_time

# --- Feature Extraction Loop ---
for label in classes:
    folder = os.path.join(dataset_path, label)
    for filename in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            y = y / np.max(np.abs(y))  # Normalize amplitude

            # --- Core Features ---
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)

            # Delta & Delta-Delta MFCCs
            mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
            mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            rms = np.mean(librosa.feature.rms(y=y))
            bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
            contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
            chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
            decay = compute_reverb_decay(y, sr)

            # --- Combine all features ---
            feature_vector = np.concatenate([
                mfccs_mean,
                mfcc_delta,
                mfcc_delta2,
                [zcr, spec_centroid, spec_flatness, rolloff, rms, bandwidth, contrast, decay],
                chroma
            ])

            features.append(feature_vector)
            labels.append(label)

# --- Convert to arrays ---
features = np.array(features)
labels = np.array(labels)

# --- Encode Labels ---
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# --- Normalize Features ---
scaler = StandardScaler()
normalized_features = scaler.fit_transform(features)

# --- Save Outputs ---
joblib.dump((normalized_features, encoded_labels, label_encoder), 'audio_features_augmented.pkl')
joblib.dump(scaler, 'scaler.pkl')  # Save scaler for inference

print("✅ All features extracted, normalized, and saved successfully.")
