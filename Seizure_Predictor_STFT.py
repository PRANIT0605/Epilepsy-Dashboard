import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, stft, welch
from scipy.fft import fft
from scipy.stats import entropy
import joblib
from io import BytesIO

# Load model and scaler
model = joblib.load("random_forest_stft_model_simple.joblib")
scaler = joblib.load("scaler_stft_simple.joblib")

# Class label mapping
class_labels = {
    0: 'Z (Healthy, Eyes Open)',
    1: 'O (Healthy, Eyes Closed)',
    2: 'N (Interictal Hippocampal)',
    3: 'F (Interictal Epileptogenic)',
    4: 'S (Seizure - Ictal)'
}

# Process EEG: STFT + band power features
def process_file(file_content):
    df = pd.read_csv(file_content)
    amplitudes = df['Amplitude (µV)'].values
    fs = 173.61
    lowcut = 0.5
    highcut = 40.0
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(2, [low, high], btype='band') # type: ignore
    filtered = filtfilt(b, a, amplitudes)

    # STFT
    f, t, Zxx = stft(filtered, fs=fs, nperseg=87, noverlap=65)
    stft_mag = np.abs(Zxx)

    # Band Power Features
    features = []
    freq_bands = [(0, 4), (4, 8), (8, 12), (12, 30), (30, 40)]
    for band in freq_bands:
        mask = (f >= band[0]) & (f <= band[1])
        band_power = np.mean(stft_mag[mask], axis=0)
        flat = band_power.flatten()
        features.extend([
            np.mean(flat),
            np.std(flat),
            np.max(flat),
            np.ptp(flat),
            entropy(flat)
        ])
    
    features = np.array(features)
    return features, filtered, f, t, Zxx

# UI Layout
st.set_page_config(page_title="EEG Seizure Detection", layout="centered")
st.title("EEG Seizure Classification (Z/O/N/F/S)")
st.write("Upload EEG CSV and classify it into one of 5 brain states.")

# Upload
uploaded_file = st.file_uploader("Upload EEG CSV File", type="csv")

# Session state initialization
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.proba = None
    st.session_state.features = None
    st.session_state.filtered = None
    st.session_state.f = None
    st.session_state.t = None
    st.session_state.Zxx = None

# Analyze
if uploaded_file and st.button("Analyze Now"):
    features, filtered, f, t, Zxx = process_file(uploaded_file)
    X = np.array([features])
    X_scaled = scaler.transform(X)

    pred = model.predict(X_scaled)[0]
    proba = model.predict_proba(X_scaled)[0]

    # Store in session state
    st.session_state.prediction = pred
    st.session_state.proba = proba
    st.session_state.features = features
    st.session_state.filtered = filtered
    st.session_state.f = f
    st.session_state.t = t
    st.session_state.Zxx = Zxx

# Show results if available
if st.session_state.prediction is not None:
    pred = st.session_state.prediction
    proba = st.session_state.proba
    features = st.session_state.features
    filtered = st.session_state.filtered
    f = st.session_state.f
    t = st.session_state.t
    Zxx = st.session_state.Zxx

    st.markdown("""
    <style>
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        font-family: 'Arial', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

    predicted_label = class_labels[pred]
    if pred == 4:
        st.error("Seizure Activity Detected (Class S)")
    else:
        st.success(f"No Seizure Detected ({predicted_label})")
    st.markdown(f'<div class="big-font">Prediction: {predicted_label}</div>', unsafe_allow_html=True)

    st.markdown("### Class Probabilities:")
    for i, label in class_labels.items():
        st.write(f"**{label}:** `{proba[i]:.4f}`") # type: ignore

    # ➤ Plot based on selection with download button
    st.markdown("---")
    st.markdown("### Choose a Visualization:")
    choice = st.selectbox(
        "View EEG Feature Representation:",
        ["FFT Plot", "STFT Spectrogram", "Power Spectral Density (PSD)", "Band Power Bar Chart", "Filtered Signal"]
    )

    if choice == "FFT Plot":              
        fft_vals = np.abs(fft(filtered))[:len(filtered)//2]
        fig_fft, ax_fft = plt.subplots()
        ax_fft.plot(fft_vals, color='darkgreen')
        ax_fft.set_title("FFT of EEG Signal")
        ax_fft.set_xlabel("Frequency Bin")
        ax_fft.set_ylabel("Amplitude")
        st.pyplot(fig_fft)
        buf = BytesIO()
        fig_fft.savefig(buf, format="png")
        st.download_button("Download FFT Plot", buf.getvalue(), file_name="fft_plot.png", mime="image/png")

    elif choice == "STFT Spectrogram":
        fig_stft, ax_stft = plt.subplots()
        ax_stft.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='magma')
        ax_stft.set_title("STFT Spectrogram")
        ax_stft.set_xlabel("Time [s]")
        ax_stft.set_ylabel("Frequency [Hz]")
        st.pyplot(fig_stft)
        buf = BytesIO()
        fig_stft.savefig(buf, format="png")
        st.download_button("Download STFT Spectrogram", buf.getvalue(), file_name="stft_spectrogram.png", mime="image/png")

    elif choice == "Power Spectral Density (PSD)":
        f_psd, psd = welch(filtered, fs=173.61, nperseg=256)
        fig_psd, ax_psd = plt.subplots()
        ax_psd.semilogy(f_psd, psd)
        ax_psd.set_title("Power Spectral Density")
        ax_psd.set_xlabel("Frequency [Hz]")
        ax_psd.set_ylabel("Power")
        st.pyplot(fig_psd)
        buf = BytesIO()
        fig_psd.savefig(buf, format="png")
        st.download_button("Download PSD Plot", buf.getvalue(), file_name="psd_plot.png", mime="image/png")

    elif choice == "Band Power Bar Chart":
        f_psd, psd = welch(filtered, fs=173.61, nperseg=256)
        bands = {'Delta': (0, 4), 'Theta': (4, 8), 'Alpha': (8, 12), 'Beta': (12, 30), 'Gamma': (30, 40)}
        powers = []
        for name, (low, high) in bands.items():
                idx = (f_psd >= low) & (f_psd <= high)
                band_power = np.trapz(psd[idx], f_psd[idx])
                powers.append(band_power)

        fig_bar, ax_bar = plt.subplots()
        ax_bar.bar(bands.keys(), powers, color='teal')
        ax_bar.set_title("EEG Band Power")
        ax_bar.set_ylabel("Power")
        st.pyplot(fig_bar)

        buf = BytesIO()
        fig_bar.savefig(buf, format="png")
        st.download_button("Download Band Power Chart", buf.getvalue(), file_name="band_power_chart.png", mime="image/png")

    elif choice == "Filtered Signal":
        fig_time, ax_time = plt.subplots()
        ax_time.plot(filtered, color='orange')
        ax_time.set_title("Filtered EEG Signal")
        ax_time.set_xlabel("Sample Index")
        ax_time.set_ylabel("Amplitude (µV)")
        st.pyplot(fig_time)

        buf = BytesIO()
        fig_time.savefig(buf, format="png")
        st.download_button("Download Filtered Signal Plot", buf.getvalue(), file_name="filtered_signal.png", mime="image/png")
