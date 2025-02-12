import streamlit as st
import tempfile
import numpy as np
import librosa
import joblib
import plotly.express as px
import matplotlib.pyplot as plt

from preprocess_function import extract_audio, preprocess_audio, predict_personality



# Streamlit app
st.title("AI Interview - Automation Personality Perception Dashboard")

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
        personality_le = joblib.load('personality_model/personality_label_encoder.joblib') #############change path################
        personality_scaler = joblib.load('personality_model/personality_feature_scaler.joblib') #############change path################ 

        # Predict personality traits
        personality_results = predict_personality(features, personality_scaler, personality_le)


    # Create tabs for Personality Traits and Emotions Analysis
    tab1, tab2 = st.tabs(["Personality Traits Analysis", "Extracted Audio Features"])

    with tab1:
        # Display Personality Traits results
        st.header("Personality Traits Analysis")
        for model, scores in personality_results.items():
            st.subheader(model)
            if len(personality_le.classes_) == len(scores):
                for trait, score in zip(personality_le.classes_, scores):
                    st.write(f"{trait}: {score * 100:.2f}%")
               

                # Plot radar charts with color fill
                fig = px.line_polar(
                    r=scores,
                    theta=personality_le.classes_,
                    line_close=True
                )
                fig.update_traces(fill='toself')
                st.plotly_chart(fig)

                # Determine the most likely personality trait
                most_likely_trait = personality_le.classes_[np.argmax(scores)]
                st.write(f"Results : The model predicts that this candidate is more likely to be {most_likely_trait}.")

                # Provide detailed description based on the most likely personality trait
                trait_descriptions = {
                    "openness": "This candidate are likely to prefer new, exciting situations. Candidate value knowledge, and friends and family are likely to describe them as curious and intellectual.",
                    "conscientiousness": "This candidate have a lot of self-discipline and exceed others’ expectations. Candidate have a strong focus and prefer order and planned activities over spontaneity.",
                    "extroversion": "Extroverts thrive in social situations. This candidate are action-oriented and appreciate the opportunity to work with others.",
                    "agreeableness": "This candidate are considerate, kind and sympathetic to others. Like to participate in group activities since they are capable at compromise and helping others.",
                    "neuroticism": "indicates anxiety and pessimism,lower means emotional stability. This measure can mean you have a more hopeful view of your circumstances. As a low-neurotic or emotionally stable person, others likely admire your calmness and resilience during challenges."
                }
                st.write(trait_descriptions.get(most_likely_trait, "No description available for this trait."))

                # Display confidence and consistency metrics
                confidence = np.max(scores) * 100
                consistency = np.std(scores) * 100
                st.write(f"Confidence: {confidence:.2f}%")
                st.write(f"Consistency: {consistency:.2f}%")
            else:
                st.error("Mismatch between traits and scores. Check model output!")

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