import streamlit as st
import pandas as pd
import numpy as np
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(layout="wide")
st.title("EEG FFT Visualizer & Comparator")


uploaded_files = st.file_uploader("Upload EEG CSV file(s)", type="csv", accept_multiple_files=True)

def compute_fft(signal, fs):
    N = len(signal)
    yf = np.abs(fft(signal))[:N // 2]
    xf = fftfreq(N, 1 / fs)[:N // 2]
    return xf, yf
    
def get_peak_freq_and_band(signal, fs):
    from scipy.fftpack import fft, fftfreq
    n = len(signal)
    freqs = fftfreq(n, 1/fs)
    fft_result = fft(signal)
    fft_vals = np.abs(fft_result)[:n // 2]
    freqs = freqs[:n // 2]

    peak_freq = freqs[np.argmax(fft_vals)]

    # Determine dominant band
    if 0.5 <= peak_freq < 4:
        band = "Delta (0.5-4 Hz)"
    elif 4 <= peak_freq < 8:
        band = "Theta (4-8 Hz)"
    elif 8 <= peak_freq < 12:
        band = "Alpha (8-12 Hz)"
    elif 12 <= peak_freq < 30:
        band = "Beta (12-30 Hz)"
    elif 30 <= peak_freq <= 45:
        band = "Gamma (30-45 Hz)"
    else:
        band = "Outside standard EEG bands"

    return round(peak_freq, 2), band

if uploaded_files:
    file_names = [file.name for file in uploaded_files]

    if len(uploaded_files) == 1:
        st.success("Single file uploaded. You can visualize or compare two channels within this file.")

        df = pd.read_csv(uploaded_files[0])
        channels = df.columns.tolist()

        ch1 = st.selectbox("Select Channel 1", channels, index=0)
        ch2 = st.selectbox("Select Channel 2", channels, index=1)

        sig1 = df[ch1].values
        sig2 = df[ch2].values
        fs = 256

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"FFT of {ch1}")
            xf1, yf1 = compute_fft(sig1, fs)
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.plot(xf1, yf1, color='navy')
            ax1.set_title("FFT Spectrum")
            ax1.set_xlabel("Frequency (Hz)")
            ax1.set_ylabel("Amplitude")
            ax1.grid(True)
            st.pyplot(fig1)
            peak1, band1 = get_peak_freq_and_band(sig1, fs)
            st.markdown(f"**Peak Frequency:** `{peak1} Hz`")
            st.markdown(f"**Dominant Band:** `{band1}`")
            buf = BytesIO()
            fig1.savefig(buf, format="png")
            st.download_button("Download FFT Plot 1", buf.getvalue(), file_name="fft_plot1.png", mime="image1/png")

        with col2:
            st.subheader(f"FFT of {ch2}")
            xf2, yf2 = compute_fft(sig2, fs)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(xf2, yf2, color='darkred')
            ax2.set_title("FFT Spectrum")
            ax2.set_xlabel("Frequency (Hz)")
            ax2.set_ylabel("Amplitude")
            ax2.grid(True)
            st.pyplot(fig2)
            peak2, band2 = get_peak_freq_and_band(sig2, fs)
            st.markdown(f"**Peak Frequency:** `{peak2} Hz`")
            st.markdown(f"**Dominant Band:** `{band2}`")
            buf = BytesIO()
            fig2.savefig(buf, format="png")
            st.download_button("Download FFT Plot 2", buf.getvalue(), file_name="fft_plot2.png", mime="image2/png")
    
           



 