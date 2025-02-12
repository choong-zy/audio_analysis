import streamlit as st
import tempfile
import numpy as np
import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from preprocess_function import extract_audio, preprocess_audio, predict_emotion




# Streamlit app
st.title("AI Interview - Speech Emotion Recognition Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    # Display video
    st.video(tfile.name)

    # Show a processing message
    with st.spinner('Processing...'):
        # Extract audio from video
        audiofile = extract_audio(tfile.name)

        # Preprocess video
        features = preprocess_audio(audiofile)

        # Load the label encoder and feature scaler
        emotion_le = joblib.load('emotion_model/emotion_label_encoder.joblib')
        emotion_scaler = joblib.load('emotion_model/emotion_feature_scaler.joblib')
        
        # Predict emotions
        emotion_results = predict_emotion(features, emotion_scaler, emotion_le)

    # Create tabs for Personality Traits and Emotions Analysis
    tab1, tab2 = st.tabs(["Emotions Analysis", "Extracted Audio Features"])

    with tab1:
        # Display Emotions results
        st.header("Emotions Analysis")
        for model, scores in emotion_results.items():
            st.subheader(model)
            if len(emotion_le.classes_) == len(scores):
                for emotion, score in zip(emotion_le.classes_, scores):
                    st.write(f"{emotion}: {score * 100:.2f}%")

                # Plot pie chart with consistent colors
                fig = px.pie(values=[s * 100 for s in scores], names=emotion_le.classes_, title=f"{model} Emotions",
                             color=emotion_le.classes_, color_discrete_sequence=px.colors.qualitative.Plotly)
                st.plotly_chart(fig)

                most_likely_emotion = emotion_le.classes_[np.argmax(scores)]
                st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_emotion}.")

                # Display confidence and consistency metrics
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
            else:
                st.error("Mismatch between emotions and scores. Check model output!")


    with tab2:
        # Display the extracted audio features
        st.header("Extracted Audio Features")
        st.write(features)
        st.write("Shape of the features:", features.shape)

        # Visualize the audio waveform
        st.header("Audio Waveform")
        y, sr = librosa.load(audiofile, sr=None)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title='Waveform of the Audio')
        st.pyplot(fig)

        # Visualize the spectrogram
        st.header("Spectrogram")
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax)
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        st.pyplot(fig)
