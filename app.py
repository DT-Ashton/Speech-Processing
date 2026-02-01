import librosa
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.title("Audio Feature Extraction and Visualization")
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "mp4", "flac"])
if uploaded_file is not None:
    # Load audio file
    y, sr = librosa.load(uploaded_file, sr=None)

    # Show audio player
    st.audio(uploaded_file)
    
    # Extract features
    stft = np.abs(librosa.stft(y))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    pitch = librosa.yin(y=y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), frame_length=2048, hop_length=512)
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=2048, hop_length=512)
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr, n_bands=6, n_fft=2048, hop_length=512)
    tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr, hop_length=512)

    # Display features
    st.subheader("Waveform")
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    st.pyplot(plt)
    
    st.subheader("Spectrogram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max), sr=sr, y_axis='log', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    st.pyplot(plt)

    st.subheader("MFCC")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    st.pyplot(plt)

    st.subheader("Pitch")
    plt.figure(figsize=(10, 4))
    plt.plot(pitch)
    plt.title('Pitch')
    plt.xlabel('Time (frames)')
    plt.ylabel('Frequency (Hz)')
    st.pyplot(plt)
    
    st.subheader("Chroma Feature")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    plt.colorbar()
    plt.title('Chroma Feature')
    st.pyplot(plt)
    
    st.subheader("Mel Spectrogram")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(mel, ref=np.max), sr=sr, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    st.pyplot(plt)

    st.subheader("Zero Crossing Rate")
    plt.figure(figsize=(10, 4))
    plt.plot(zcr[0])
    plt.title('Zero Crossing Rate')
    plt.xlabel('Time (frames)')
    plt.ylabel('ZCR')
    st.pyplot(plt)
    
    st.subheader("Spectral Contrast")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(contrast, x_axis='time')
    plt.colorbar()
    plt.title('Spectral Contrast')
    st.pyplot(plt)
    
    st.subheader("Tonnetz")
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(tonnetz, y_axis='tonnetz', x_axis='time')
    plt.colorbar()
    plt.title('Tonnetz')
    st.pyplot(plt)
