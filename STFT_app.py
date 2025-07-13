import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import stft
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("EEG STFT Visualizer & Comparator")

uploaded_files = st.file_uploader("Upload EEG CSV file(s)", type="csv", accept_multiple_files=True)

if uploaded_files:
    file_names = [file.name for file in uploaded_files]

    if len(uploaded_files) == 1:
        # Single-file mode
        st.success("Single file uploaded. You can visualize or compare two channels within this file.")

        df = pd.read_csv(uploaded_files[0])
        channels = df.columns.tolist()

        # Channel selection
        ch1 = st.selectbox("Select Channel 1", channels, index=0)
        ch2 = st.selectbox("Select Channel 2 (for comparison)", channels, index=1)

        sig1 = df[ch1].values
        sig2 = df[ch2].values
        fs = 256

        f1, t1, Zxx1 = stft(sig1, fs=fs, nperseg=128)
        f2, t2, Zxx2 = stft(sig2, fs=fs, nperseg=128)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"STFT of {ch1}")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.pcolormesh(t1, f1, np.abs(Zxx1), shading='gouraud')
            ax1.set_title(f"{ch1} Spectrogram")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_xlabel("Time (s)")
            st.pyplot(fig1)

        with col2:
            st.subheader(f"STFT of {ch2}")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.pcolormesh(t2, f2, np.abs(Zxx2), shading='gouraud')
            ax2.set_title(f"{ch2} Spectrogram")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (s)")
            st.pyplot(fig2)

    else:
        # Multi-file comparison mode
        st.success(f"{len(uploaded_files)} files uploaded. Compare channels across patients.")

        file1_name = st.selectbox("Select File 1", file_names, index=0, key="file1")
        file2_name = st.selectbox("Select File 2", file_names, index=1, key="file2")

        df1 = pd.read_csv([f for f in uploaded_files if f.name == file1_name][0])
        df2 = pd.read_csv([f for f in uploaded_files if f.name == file2_name][0])

        ch1 = st.selectbox(f"Channel from {file1_name}", df1.columns.tolist(), key="ch1")
        ch2 = st.selectbox(f"Channel from {file2_name}", df2.columns.tolist(), key="ch2")

        sig1 = df1[ch1].values
        sig2 = df2[ch2].values
        fs = 256

        f1, t1, Zxx1 = stft(sig1, fs=fs, nperseg=128)
        f2, t2, Zxx2 = stft(sig2, fs=fs, nperseg=128)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"{file1_name} - {ch1}")
            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.pcolormesh(t1, f1, np.abs(Zxx1), shading='gouraud')
            ax1.set_title("STFT Spectrogram")
            ax1.set_ylabel("Frequency (Hz)")
            ax1.set_xlabel("Time (s)")
            st.pyplot(fig1)

        with col2:
            st.subheader(f"{file2_name} - {ch2}")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.pcolormesh(t2, f2, np.abs(Zxx2), shading='gouraud')
            ax2.set_title("STFT Spectrogram")
            ax2.set_ylabel("Frequency (Hz)")
            ax2.set_xlabel("Time (s)")
            st.pyplot(fig2)

else:
    st.warning("Upload at least one CSV EEG file to proceed.")
 