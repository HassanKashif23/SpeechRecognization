import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pydub import AudioSegment
import tempfile
import os

# Load trained model and label encoder
model = joblib.load('trained_model.joblib')
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')

# --- Enhanced Feature Extraction Function ---
def extract_features(audio, sr):
    audio = audio / np.max(np.abs(audio))
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # ZCR
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Spectral Flatness (proxy for entropy)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    spec_flatness_mean = np.mean(spec_flatness)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = np.mean(rolloff)

    # RMS
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    # Reverb Decay Time (approximated)
    energy = rms[0]
    peak_idx = np.argmax(energy)
    peak_energy = energy[peak_idx]
    decay_threshold = peak_energy * 0.1
    decay_time = 0.0
    for i in range(peak_idx + 1, len(energy)):
        if energy[i] < decay_threshold:
            decay_time = (i - peak_idx) * (512 / sr)  # hop_length = 512
            break

    # Combine all features
    feature_vector = np.hstack([
        mfcc_mean,
        zcr_mean,
        spec_centroid_mean,
        spec_flatness_mean,
        rolloff_mean,
        rms_mean,
        decay_time
    ])
    return feature_vector.reshape(1, -1)

# --- Streamlit App UI ---
st.set_page_config(page_title="Audio Analyzer", layout="centered")
st.title("🔊 Audio Analyzer")
st.write("Upload an audio file to classify whether it's a **safe (direct call)** or **unauthorized (speakerphone)** recording.")

uploaded_file = st.file_uploader("📤 Upload your audio file", type=["wav", "mp3", "m4a", "ogg"])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        file_path = tmpfile.name
        if not uploaded_file.name.lower().endswith(".wav"):
            # Convert to WAV using pydub
            audio = AudioSegment.from_file(uploaded_file)
            audio.export(file_path, format="wav")
        else:
            tmpfile.write(uploaded_file.read())

    # Load and extract features
    x, sr = librosa.load(file_path, sr=None)
    features = extract_features(x, sr)

    # Predict
    pred_proba = model.predict_proba(features)[0]
    pred_index = np.argmax(pred_proba)
    predicted_class = label_encoder.inverse_transform([pred_index])[0]

    # Display result
    st.markdown(f"### 🟩 Predicted Class: `{predicted_class}`")

    st.subheader("📊 Confidence Scores:")
    for i, class_name in enumerate(label_encoder.classes_):
        st.write(f"- **{class_name}**: {pred_proba[i]*100:.2f}%")

    st.bar_chart({label: pred_proba[i] for i, label in enumerate(label_encoder.classes_)})

    # Cleanup
    os.remove(file_path)
