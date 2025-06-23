import streamlit as st
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Load trained model and label encoder
model = joblib.load('trained_model.joblib')
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')

# --- Feature extraction function ---
def extract_features(audio, sr):
    # Normalize
    audio = audio / np.max(np.abs(audio))
    
    # Resample to 16k if needed
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Feature 1: MFCCs (13)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)

    # Feature 2: Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = np.mean(zcr)

    # Feature 3: Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    spec_centroid_mean = np.mean(spec_centroid)

    # Feature 4: Spectral Entropy (approximated using spectral flatness)
    spec_flatness = librosa.feature.spectral_flatness(y=audio)
    spec_flatness_mean = np.mean(spec_flatness)

    # Feature 5: Spectral Roll-off
    rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    rolloff_mean = np.mean(rolloff)

    # Feature 6: RMS Energy
    rms = librosa.feature.rms(y=audio)
    rms_mean = np.mean(rms)

    # Combine all features (order matters!)
    feature_vector = np.hstack([
        mfcc_mean,
        zcr_mean,
        spec_centroid_mean,
        spec_flatness_mean,
        rolloff_mean,
        rms_mean
    ])

    return feature_vector.reshape(1, -1)

# --- Streamlit App UI ---
st.set_page_config(page_title="Audio Safety Classifier", layout="centered")
st.title("🔊 Audio Safety Classifier")
st.write("Upload a `.wav` file to classify whether it's a **safe (direct call)** or **unsafe (speakerphone)** recording.")

uploaded_file = st.file_uploader("📤 Upload your `.wav` file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    
    # Load and preprocess
    x, sr = librosa.load(uploaded_file, sr=None)
    features = extract_features(x, sr)

    # Predict
    pred_proba = model.predict_proba(features)[0]
    pred_index = np.argmax(pred_proba)
    predicted_class = label_encoder.inverse_transform([pred_index])[0]

    # Show prediction
    st.markdown(f"### 🟩 Predicted Class: `{predicted_class}`")
    
    # Show confidence
    st.subheader("📊 Confidence Scores:")
    for i, class_name in enumerate(label_encoder.classes_):
        st.write(f"- **{class_name}**: {pred_proba[i]*100:.2f}%")

    # Optional: Bar chart
    st.bar_chart({label: pred_proba[i] for i, label in enumerate(label_encoder.classes_)})
