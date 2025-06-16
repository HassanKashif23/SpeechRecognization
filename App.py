import streamlit as st
import librosa
import numpy as np
import joblib

# Load trained model and label encoder
model = joblib.load('trained_model.joblib')
_, _, label_encoder = joblib.load('audio_features.pkl')

st.set_page_config(page_title="Audio Classifier", layout="centered")

st.title("🎧 Audio Classification App")
st.markdown("Upload a `.wav` file and the model will predict if it's **Conversation** or **Background Noise**.")

# File uploader
uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    if st.button("🔍 Predict"):
        try:
            # Load audio from uploaded file
            x, fs = librosa.load(uploaded_file, sr=None)

            # Normalize
            x = x / np.max(np.abs(x))

            # Resample if needed
            if fs != 16000:
                x = librosa.resample(x, orig_sr=fs, target_sr=16000)
                fs = 16000

            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13)
            mean_features = np.mean(mfccs.T, axis=0).reshape(1, -1)

            # Predict
            pred_encoded = model.predict(mean_features)[0]
            predicted_class = label_encoder.inverse_transform([pred_encoded])[0]

            st.success(f"🟩 Predicted Class: **{predicted_class}**")

        except Exception as e:
            st.error(f"❌ Error processing the file: {e}")
