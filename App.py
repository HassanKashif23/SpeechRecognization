import streamlit as st
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import tempfile
import os

# --- Load trained model, scaler, and label encoder ---
model = joblib.load('trained_model.joblib')
scaler = joblib.load('scaler.pkl')
_, _, label_encoder = joblib.load('audio_features_augmented.pkl')


# --- Feature Extraction Function ---
def extract_features(audio, sr):

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        sr = 16000

    # MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # Delta & Delta-Delta
    mfcc_delta = np.mean(librosa.feature.delta(mfccs), axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

    # Spectral features
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    spec_flatness = np.mean(librosa.feature.spectral_flatness(y=audio))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
    rms = np.mean(librosa.feature.rms(y=audio))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr))

    # Chroma
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr), axis=1)

    # Reverb decay
    energy = librosa.feature.rms(y=audio)[0]
    peak_idx = np.argmax(energy)
    peak_energy = energy[peak_idx]
    decay_threshold = peak_energy * 0.1
    decay_time = 0.0

    for i in range(peak_idx + 1, len(energy)):
        if energy[i] < decay_threshold:
            decay_time = (i - peak_idx) * (512 / sr)
            break

    # Combine features (59)
    feature_vector = np.concatenate([
        mfccs_mean,
        mfcc_delta,
        mfcc_delta2,
        [zcr, spec_centroid, spec_flatness, rolloff, rms, bandwidth, contrast, decay_time],
        chroma
    ])

    return feature_vector.reshape(1, -1)


# --- Streamlit UI ---
st.set_page_config(page_title="Audio Analyzer", layout="centered")

st.title("🔊 Audio Analyzer")
st.write("Upload an audio file to classify whether it's a **safe (direct call)** or **unauthorized (speakerphone)** recording.")

uploaded_file = st.file_uploader(
    "📤 Upload your audio file",
    type=["wav", "mp3", "m4a", "ogg"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    # -------- Save uploaded file to disk --------
    with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp:
        tmp.write(uploaded_file.read())
        input_path = tmp.name

    # -------- Convert to WAV --------
    if not input_path.lower().endswith(".wav"):

        audio = AudioSegment.from_file(input_path)

        # enforce same format as training
        audio = audio.set_frame_rate(16000).set_channels(1)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            audio.export(tmp_wav.name, format="wav")
            file_path = tmp_wav.name
    else:
        file_path = input_path

    # -------- Load audio --------
    x, sr = librosa.load(file_path, sr=None)

    # -------- Feature extraction --------
    features = extract_features(x, sr)

    # -------- Scaling --------
    scaled_features = scaler.transform(features)

    # -------- Prediction --------
    pred_proba = model.predict_proba(scaled_features)[0]
    pred_index = np.argmax(pred_proba)
    predicted_class = label_encoder.inverse_transform([pred_index])[0]

    # -------- Display result --------
    st.markdown(f"### 🟩 Predicted Class: `{predicted_class}`")

    st.subheader("📊 Confidence Scores")

    for i, class_name in enumerate(label_encoder.classes_):
        st.write(f"**{class_name}**: {pred_proba[i]*100:.2f}%")

    st.bar_chart({label: pred_proba[i] for i, label in enumerate(label_encoder.classes_)})

    # -------- Cleanup --------
    os.remove(file_path)
    os.remove(input_path)
