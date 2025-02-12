import tempfile
import numpy as np
import pandas as pd
import librosa
import parselmouth
from typing import Tuple
from moviepy.editor import VideoFileClip
import joblib
import tensorflow as tf


# Function to resample audio to a target sampling rate
def resampling(y: np.ndarray, sr: int, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    y_resampled = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y_resampled, target_sr

# Function to denoise audio using MMSE
def mmse_denoise(noisy_signal: np.ndarray) -> np.ndarray:
    stft_signal = librosa.stft(noisy_signal)
    noise_psd = np.mean(np.abs(stft_signal[:, :10]) ** 2, axis=1, keepdims=True)
    speech_psd = np.abs(stft_signal) ** 2
    gain = np.maximum(1 - noise_psd / (speech_psd + 1e-10), 0)
    enhanced_stft = gain * stft_signal
    enhanced_signal = librosa.istft(enhanced_stft)
    return enhanced_signal

# Function to normalize audio
def normalize_audio(audio: np.ndarray) -> np.ndarray:
    # Ensure the audio signal has non-zero elements to avoid division by zero
    if np.max(np.abs(audio)) == 0:
        return audio  # If the audio is silent, return it as-is
    
    # Normalize audio by dividing by the maximum absolute value
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio


# Function to get statistical summaries
def get_statistics(feature):
    if feature.size == 0 or np.all(np.isnan(feature)):
        return {"mean": 0, "min": 0, "max": 0, "std_dev": 0}
    return {
        'mean': np.nanmean(feature),
        'min': np.nanmin(feature),
        'max': np.nanmax(feature),
        'std_dev': np.nanstd(feature)
    }

# Function to extract audio features
def extract_audio_features(y: np.ndarray, sr: int) -> pd.DataFrame:
    # Load the audio into PRAAT's Sound object
    sound = parselmouth.Sound(y, sampling_frequency=sr)

     # Extract intensity and pitch objects for prosodic features
    intensity_obj = sound.to_intensity()
    pitch_obj = sound.to_pitch(pitch_floor=75, pitch_ceiling=600)
    
    # Prosodic features
    # Intensity
    intensity_values = intensity_obj.values.flatten()
    intensity_stats = get_statistics(intensity_values)

    # Pitch
    pitch_values = pitch_obj.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # Filter unvoiced frames
    pitch_stats = get_statistics(pitch_values)

    # Energy
    energy = np.sum(y ** 2)

    # Speech rate (zero crossings per second)
    zero_crossings = np.sum(np.abs(np.diff(np.sign(y))) > 0)
    duration = sound.get_total_duration()
    speech_rate = zero_crossings / duration

    # Spectral features
    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_stats = [get_statistics(mfcc[i]) for i in range(mfcc.shape[0])]

    # Chroma
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stats = [get_statistics(chroma[i]) for i in range(chroma.shape[0])]

    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    spectral_contrast_stats = [get_statistics(spectral_contrast[i]) for i in range(spectral_contrast.shape[0])]

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_stats = get_statistics(spectral_centroid[0])

    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_stats = get_statistics(spectral_bandwidth[0])

    # Spectral flatness
    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    spectral_flatness_stats = get_statistics(spectral_flatness[0])

    # Spectral roll-off
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    spectral_rolloff_stats = get_statistics(spectral_rolloff[0])

    # LPC (Linear Predictive Coding)
    lpc_coeffs = librosa.lpc(y, order=16)
    lpc_stats = get_statistics(lpc_coeffs)


     # Voice Quality features
    # Formants (F1, F2, F3)
    formant = sound.to_formant_burg()
    f1 = np.array([formant.get_value_at_time(1, t) for t in np.linspace(0, duration, 100)])
    f1_stats = get_statistics(f1)
    f2 = np.array([formant.get_value_at_time(2, t) for t in np.linspace(0, duration, 100)])
    f2_stats = get_statistics(f2)
    f3 = np.array([formant.get_value_at_time(3, t) for t in np.linspace(0, duration, 100)])
    f3_stats = get_statistics(f3)

    # Harmonic-to-Noise Ratio (HNR)
    hnr = sound.to_harmonicity()
    hnr_values = hnr.values.flatten()
    hnr_stats = get_statistics(hnr_values)

    # RMS energy
    rms = librosa.feature.rms(y=y)
    rms_stats = get_statistics(rms[0])

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_stats = get_statistics(zero_crossing_rate[0])

    # Compile features into a DataFrame
    features = {
        # Prosodic features
        'intensity_mean': intensity_stats['mean'],
        'intensity_min': intensity_stats['min'],
        'intensity_max': intensity_stats['max'],
        'intensity_std_dev': intensity_stats['std_dev'],
        'pitch_mean': pitch_stats['mean'],
        'pitch_min': pitch_stats['min'],
        'pitch_max': pitch_stats['max'],
        'pitch_std_dev': pitch_stats['std_dev'],
        'energy': energy,
        'speech_rate': speech_rate,

        # Spectral features
        'spectral_centroid_mean': spectral_centroid_stats['mean'],
        'spectral_centroid_min': spectral_centroid_stats['min'],
        'spectral_centroid_max': spectral_centroid_stats['max'],
        'spectral_centroid_std_dev': spectral_centroid_stats['std_dev'],
        'spectral_bandwidth_mean': spectral_bandwidth_stats['mean'],
        'spectral_bandwidth_min': spectral_bandwidth_stats['min'],
        'spectral_bandwidth_max': spectral_bandwidth_stats['max'],
        'spectral_bandwidth_std_dev': spectral_bandwidth_stats['std_dev'],
        'spectral_flatness_mean': spectral_flatness_stats['mean'],
        'spectral_flatness_min': spectral_flatness_stats['min'],
        'spectral_flatness_max': spectral_flatness_stats['max'],
        'spectral_flatness_std_dev': spectral_flatness_stats['std_dev'],
        'spectral_rolloff_mean': spectral_rolloff_stats['mean'],
        'spectral_rolloff_min': spectral_rolloff_stats['min'],
        'spectral_rolloff_max': spectral_rolloff_stats['max'],
        'spectral_rolloff_std_dev': spectral_rolloff_stats['std_dev'],
        **{f'mfcc_{i}_mean': mfcc_stats[i]['mean'] for i in range(13)},
        **{f'mfcc_{i}_min': mfcc_stats[i]['min'] for i in range(13)},
        **{f'mfcc_{i}_max': mfcc_stats[i]['max'] for i in range(13)},
        **{f'mfcc_{i}_std_dev': mfcc_stats[i]['std_dev'] for i in range(13)},
        **{f'chroma_{i}_mean': chroma_stats[i]['mean'] for i in range(12)},
        **{f'chroma_{i}_min': chroma_stats[i]['min'] for i in range(12)},
        **{f'chroma_{i}_std_dev': chroma_stats[i]['std_dev'] for i in range(12)},
        **{f'spectral_contrast_{i}_mean': spectral_contrast_stats[i]['mean'] for i in range(7)},
        **{f'spectral_contrast_{i}_min': spectral_contrast_stats[i]['min'] for i in range(7)},
        **{f'spectral_contrast_{i}_max': spectral_contrast_stats[i]['max'] for i in range(7)},
        **{f'spectral_contrast_{i}_std_dev': spectral_contrast_stats[i]['std_dev'] for i in range(7)},
        'lpc_mean': lpc_stats['mean'],
        'lpc_min': lpc_stats['min'],
        'lpc_max': lpc_stats['max'],
        'lpc_std_dev': lpc_stats['std_dev'],

        # Voice Quality features
        'f1_mean': f1_stats['mean'],
        'f1_min': f1_stats['min'],
        'f1_max': f1_stats['max'],
        'f1_std_dev': f1_stats['std_dev'],
        'f2_mean': f2_stats['mean'],
        'f2_min': f2_stats['min'],
        'f2_max': f2_stats['max'],
        'f2_std_dev': f2_stats['std_dev'],
        'f3_mean': f3_stats['mean'],
        'f3_min': f3_stats['min'],
        'f3_max': f3_stats['max'],
        'f3_std_dev': f3_stats['std_dev'],
        'hnr_mean': hnr_stats['mean'],
        'hnr_min': hnr_stats['min'],
        'hnr_max': hnr_stats['max'],
        'hnr_std_dev': hnr_stats['std_dev'],
        'rms_mean': rms_stats['mean'],
        'rms_min': rms_stats['min'],
        'rms_max': rms_stats['max'],
        'rms_std_dev': rms_stats['std_dev'],
        'zero_crossing_rate_mean': zero_crossing_rate_stats['mean'],
        'zero_crossing_rate_min': zero_crossing_rate_stats['min'],
        'zero_crossing_rate_max': zero_crossing_rate_stats['max'],
        'zero_crossing_rate_std_dev': zero_crossing_rate_stats['std_dev']
    }

    return pd.DataFrame([features])



# Preprocessing function for the input audio
def preprocess_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    resampled_audio, new_sr = resampling(y, sr)
    denoised_audio = mmse_denoise(resampled_audio)
    normalized_audio = normalize_audio(denoised_audio)
    features = extract_audio_features(normalized_audio, new_sr)
    return features


# Function to extract audio from video
def extract_audio(video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3").name
    audio_clip.write_audiofile(audio_path)
    return audio_path



def predict_personality(features,scaler,le):

    # Transform the features
    features = scaler.transform(features)
    features_rnn = features.reshape(features.shape[0], 1, features.shape[1])

    # Load the trained models
    rnn_model = tf.keras.models.load_model('personality_model/personality_rnn_model.h5') #############change path################
    xgb_model = joblib.load('personality_model/personality_xgboost_model.joblib') #############change path################
    rf_model = joblib.load('personality_model/personality_rf_model.joblib') #############change path################
    svm_model = joblib.load('personality_model/personality_svm_model.joblib') #############change path################
 
    # Predict personality traits using the models
    rnn_prediction = rnn_model.predict(features_rnn)
    xgb_prediction = xgb_model.predict_proba(features)
    rf_prediction = rf_model.predict_proba(features)
    svm_prediction = svm_model.predict_proba(features)

    # Soft Voting (Average) Ensemble
    final_probs = (rnn_prediction + xgb_prediction + rf_prediction + svm_prediction) / 4

    # Combine predictions into a dictionary
    predictions = {
        # 'Prediction from RNN model': rnn_prediction.flatten().tolist(),
        # 'Prediction from XGBoost model': xgb_prediction.flatten().tolist(),
        # 'Prediction from Random Forest model': rf_prediction.flatten().tolist(),
        # 'Prediction from SVM model': svm_prediction.flatten().tolist(),
        'Prediction Result from Ensemble model': final_probs.flatten().tolist()
    }
    return predictions
    

def predict_emotion(features,scaler,le):

    # Transform the features
    features = scaler.transform(features)
    features_rnn = features.reshape(features.shape[0], 1, features.shape[1])

    # Load the trained models
    rnn_model = tf.keras.models.load_model('emotion_model/emotion_rnn_model.h5') #############change path################
    xgb_model = joblib.load('emotion_model/emotion_xgboost_model.joblib') #############change path################
    rf_model = joblib.load('emotion_model/emotion_rf_model.joblib') #############change path################
    svm_model = joblib.load('emotion_model/emotion_svm_model.joblib') #############change path################

    # Predict emotion traits using the models
    rnn_prediction = rnn_model.predict(features_rnn)
    xgb_prediction = xgb_model.predict_proba(features)
    rf_prediction = rf_model.predict_proba(features)
    svm_prediction = svm_model.predict_proba(features)

    # Soft Voting (Average) Ensemble
    final_probs = (rnn_prediction + xgb_prediction + rf_prediction + svm_prediction) / 4
    

    # Combine predictions into a dictionary
    predictions = {
        # 'Prediction from RNN model': rnn_prediction.flatten().tolist(),
        # 'Prediction from XGBoost model': xgb_prediction.flatten().tolist(),
        # 'Prediction from Random Forest model': rf_prediction.flatten().tolist(),
        # 'Prediction from SVM model': svm_prediction.flatten().tolist(),
        'Prediction Result from Ensemble model': final_probs.flatten().tolist()
    }
    return predictions