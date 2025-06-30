import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tempfile
import os

# --- Load trained model, scaler, and label encoder ---
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.pkl')  # 💡 New: Apply the same scaler used in training
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')

# --- Feature Extraction Function ---
def extract_features(audio, sr):
    audio = audio / np.max(np.abs(audio))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))

    # Spectral Centroid
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))

    # Spectral Flatness
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))

    # Spectral Rolloff
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))

    # RMS
    rms = np.mean(librosa.feature.rms(y=audio))

    # Reverb Decay Time (approximated)
    energy = librosa.feature.rms(y=audio)[0]
    decay = np.polyfit(np.arange(len(energy)), 20 * np.log10(energy + 1e-10), deg=1)[0]

    # Combine all 19 features
    feature_vector = np.hstack([
        mfcc_mean,
        [zcr, spec_centroid, spec_flatness, rolloff, rms, decay]
    ])
    return feature_vector.reshape(1, -1)

# --- Streamlit UI ---
st.set_page_config(page_title="Audio Analyzer", layout="centered")
st.title("🔊 Audio Analyzer")
st.write("Upload an audio file to classify whether it's a **safe (direct call)** or **unauthorized (speakerphone)** recording.")

uploaded_file = st.file_uploader("📤 Upload your audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        file_path = tmpfile.name
        if not uploaded_file.name.lower().endswith(".wav"):
            audio = AudioSegment.from_file(uploaded_file)
            audio.export(file_path, format="wav")
        else:
            tmpfile.write(uploaded_file.read())

    # Load and extract features
    x, sr = librosa.load(file_path, sr=None)
    features = extract_features(x, sr)

    # Apply feature scaling
    scaled_features = scaler.transform(features)

    # Predict
    pred_proba = model.predict_proba(scaled_features)[0]
    pred_index = np.argmax(pred_proba)
    predicted_class = label_encoder.inverse_transform([pred_index])[0]

    # Display result
    st.markdown(f"### 🟩 Predicted Class: `{predicted_class}`")

    st.subheader("📊 Confidence Scores:")
    for i, class_name in enumerate(label_encoder.classes_):
        st.write(f"- **{class_name}**: {pred_proba[i]*100:.2f}%")

    st.bar_chart({label: pred_proba[i] for i, label in enumerate(label_encoder.classes_)})

    os.remove(file_path)
