import os
import numpy as np
import librosa
import joblib
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Set your dataset path
dataset_path = r'D:\AudioDetection\augmented_dataset'  # Replace with your actual path
classes = ['Conversation', 'Speakerphone', 'Background']

features = []
labels = []

def compute_reverb_decay(audio, sr):
    """Approximate reverb decay using RMS energy envelope"""
    rms = librosa.feature.rms(y=audio)[0]
    peak_idx = np.argmax(rms)
    peak_energy = rms[peak_idx]
    decay_threshold = peak_energy * 0.1
    decay_time = 0.0
    for i in range(peak_idx + 1, len(rms)):
        if rms[i] < decay_threshold:
            decay_time = (i - peak_idx) * (512 / sr)  # hop_length = 512
            break
    return decay_time

# Loop through all classes and extract features
for label in classes:
    folder = os.path.join(dataset_path, label)
    for filename in tqdm(os.listdir(folder), desc=f"Processing {label}"):
        if filename.endswith('.wav'):
            filepath = os.path.join(folder, filename)
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            y = y / np.max(np.abs(y))

            # Feature 1: MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfccs_mean = np.mean(mfccs, axis=1)

            # Feature 2: Zero Crossing Rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))

            # Feature 3: Spectral Centroid
            spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

            # Feature 4: Spectral Flatness
            spec_flatness = np.mean(librosa.feature.spectral_flatness(y=y))

            # Feature 5: Spectral Rolloff
            rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

            # Feature 6: RMS Energy
            rms = np.mean(librosa.feature.rms(y=y))

            # Feature 7: Reverb Decay
            decay = compute_reverb_decay(y, sr)

            # Combine all features (19 total)
            feature_vector = np.concatenate([
                mfccs_mean,
                [zcr, spec_centroid, spec_flatness, rolloff, rms, decay]
            ])
            features.append(feature_vector)
            labels.append(label)

# Convert and encode
features = np.array(features)
labels = np.array(labels)
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Save to file
joblib.dump((features, encoded_labels, label_encoder), 'audio_features_augmented.pkl')
print("✅ All 19 features extracted and saved successfully.")
