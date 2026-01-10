"""
Voice Deepfake and Vishing Detection Module

This module provides machine learning models for detecting voice deepfakes
and vishing (voice phishing) attempts through audio analysis and pattern recognition.
"""

import numpy as np
import librosa
import librosa.feature
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.signal import spectrogram
from scipy.fft import fft
import warnings
from typing import Tuple, Dict, List, Optional
import pickle
import os

warnings.filterwarnings('ignore')


class AudioFeatureExtractor:
    """
    Extracts relevant audio features for voice deepfake and vishing detection.
    """

    def __init__(self, sr: int = 16000):
        """
        Initialize the feature extractor.

        Args:
            sr (int): Sampling rate for audio processing
        """
        self.sr = sr
        self.scaler = StandardScaler()

    def extract_mfcc_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract Mel-Frequency Cepstral Coefficients (MFCC).

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: MFCC features (mean and std of 13 coefficients)
        """
        mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        return np.concatenate([mfcc_mean, mfcc_std])

    def extract_spectral_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract spectral features from audio signal.

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Spectral features including centroid, rolloff, zero crossing rate
        """
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)
        centroid_mean = np.mean(spectral_centroid)
        centroid_std = np.std(spectral_centroid)

        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)
        rolloff_mean = np.mean(spectral_rolloff)
        rolloff_std = np.std(spectral_rolloff)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        return np.array([
            centroid_mean, centroid_std,
            rolloff_mean, rolloff_std,
            zcr_mean, zcr_std
        ])

    def extract_chromagram_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract chromagram features for pitch-related analysis.

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Chromagram statistics
        """
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sr)
        chroma_mean = np.mean(chroma, axis=1)
        chroma_std = np.std(chroma, axis=1)
        return np.concatenate([chroma_mean, chroma_std])

    def extract_tempogram_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract tempo and rhythm-related features.

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Tempo and rhythm features
        """
        # Onset strength
        onset_env = librosa.onset.onset_strength(y=y, sr=self.sr)
        onset_mean = np.mean(onset_env)
        onset_std = np.std(onset_env)

        # Tempogram
        tempogram = librosa.feature.tempogram(y=y, sr=self.sr)
        tempo_mean = np.mean(tempogram)
        tempo_std = np.std(tempogram)

        return np.array([onset_mean, onset_std, tempo_mean, tempo_std])

    def extract_prosody_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract prosody features (pitch, energy patterns).

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Prosody features
        """
        # RMS Energy
        rms = librosa.feature.rms(y=y)
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        rms_max = np.max(rms)
        rms_min = np.min(rms)

        # Spectral flux (measures changes in spectrum over time)
        spec = np.abs(librosa.stft(y))
        spec_flux = np.sqrt(np.sum(np.diff(spec, axis=1) ** 2, axis=0))
        flux_mean = np.mean(spec_flux)
        flux_std = np.std(spec_flux)

        return np.array([
            rms_mean, rms_std, rms_max, rms_min,
            flux_mean, flux_std
        ])

    def extract_cepstral_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract cepstral features for voice authenticity.

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Cepstral features
        """
        # Mel-scale spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=self.sr)
        S_db = librosa.power_to_db(S, ref=np.max)

        cepstral_mean = np.mean(S_db, axis=1)
        cepstral_std = np.std(S_db, axis=1)

        return np.concatenate([cepstral_mean, cepstral_std])

    def extract_all_features(self, y: np.ndarray) -> np.ndarray:
        """
        Extract all available features from audio.

        Args:
            y (np.ndarray): Audio time series

        Returns:
            np.ndarray: Combined feature vector
        """
        mfcc = self.extract_mfcc_features(y)
        spectral = self.extract_spectral_features(y)
        chroma = self.extract_chromagram_features(y)
        tempogram = self.extract_tempogram_features(y)
        prosody = self.extract_prosody_features(y)
        cepstral = self.extract_cepstral_features(y)

        return np.concatenate([mfcc, spectral, chroma, tempogram, prosody, cepstral])

    def load_and_process_audio(self, audio_path: str) -> np.ndarray:
        """
        Load and process audio file.

        Args:
            audio_path (str): Path to audio file

        Returns:
            np.ndarray: Audio time series
        """
        y, _ = librosa.load(audio_path, sr=self.sr)
        return y


class DeepfakeDetectionModel:
    """
    Machine learning model for detecting voice deepfakes.
    """

    def __init__(self):
        """Initialize the deepfake detection model."""
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the deepfake detection model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels (0: authentic, 1: deepfake)

        Returns:
            Dict[str, float]: Training metrics
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        train_score = self.model.score(X_scaled, y_train)
        return {'train_accuracy': train_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict deepfake probability for audio samples.

        Args:
            X (np.ndarray): Feature vectors

        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_from_audio(self, audio_path: str) -> Tuple[float, Dict]:
        """
        Predict deepfake probability from audio file.

        Args:
            audio_path (str): Path to audio file

        Returns:
            Tuple[float, Dict]: Deepfake score and analysis details
        """
        y = self.feature_extractor.load_and_process_audio(audio_path)
        features = self.feature_extractor.extract_all_features(y)
        deepfake_score = self.predict(features.reshape(1, -1))[0]

        analysis = {
            'deepfake_score': float(deepfake_score),
            'is_deepfake': bool(deepfake_score > 0.5),
            'confidence': float(max(deepfake_score, 1 - deepfake_score)),
            'audio_duration': len(y) / self.feature_extractor.sr
        }

        return deepfake_score, analysis


class VishingDetectionModel:
    """
    Machine learning model for detecting vishing (voice phishing) attempts.
    """

    def __init__(self):
        """Initialize the vishing detection model."""
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the vishing detection model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels (0: legitimate, 1: vishing)

        Returns:
            Dict[str, float]: Training metrics
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        train_score = self.model.score(X_scaled, y_train)
        return {'train_accuracy': train_score}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict vishing probability for audio samples.

        Args:
            X (np.ndarray): Feature vectors

        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def predict_from_audio(self, audio_path: str) -> Tuple[float, Dict]:
        """
        Predict vishing probability from audio file.

        Args:
            audio_path (str): Path to audio file

        Returns:
            Tuple[float, Dict]: Vishing score and analysis details
        """
        y = self.feature_extractor.load_and_process_audio(audio_path)
        features = self.feature_extractor.extract_all_features(y)
        vishing_score = self.predict(features.reshape(1, -1))[0]

        analysis = {
            'vishing_score': float(vishing_score),
            'is_vishing': bool(vishing_score > 0.5),
            'confidence': float(max(vishing_score, 1 - vishing_score)),
            'audio_duration': len(y) / self.feature_extractor.sr,
            'threat_level': classify_threat_level(vishing_score)
        }

        return vishing_score, analysis


class VoiceAuthenticationModel:
    """
    Model for voice authentication and speaker verification.
    """

    def __init__(self):
        """Initialize the voice authentication model."""
        self.model = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=10,
            random_state=42
        )
        self.feature_extractor = AudioFeatureExtractor()
        self.scaler = StandardScaler()
        self.is_trained = False
        self.reference_features = None

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train the voice authentication model.

        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels (0: unauthorized, 1: authorized)

        Returns:
            Dict[str, float]: Training metrics
        """
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_trained = True

        train_score = self.model.score(X_scaled, y_train)
        return {'train_accuracy': train_score}

    def enroll_speaker(self, audio_paths: List[str]) -> np.ndarray:
        """
        Enroll a speaker by extracting and averaging features.

        Args:
            audio_paths (List[str]): List of audio files for enrollment

        Returns:
            np.ndarray: Speaker reference features
        """
        features_list = []
        for path in audio_paths:
            y = self.feature_extractor.load_and_process_audio(path)
            features = self.feature_extractor.extract_all_features(y)
            features_list.append(features)

        self.reference_features = np.mean(features_list, axis=0)
        return self.reference_features

    def verify_speaker(self, audio_path: str, threshold: float = 0.7) -> Dict:
        """
        Verify if audio matches enrolled speaker.

        Args:
            audio_path (str): Path to audio file to verify
            threshold (float): Similarity threshold

        Returns:
            Dict: Verification results
        """
        if self.reference_features is None:
            raise ValueError("No speaker enrolled. Call enroll_speaker first.")

        y = self.feature_extractor.load_and_process_audio(audio_path)
        features = self.feature_extractor.extract_all_features(y)

        # Calculate cosine similarity
        similarity = np.dot(features, self.reference_features) / (
            np.linalg.norm(features) * np.linalg.norm(self.reference_features)
        )

        return {
            'similarity_score': float(similarity),
            'is_verified': bool(similarity > threshold),
            'threshold': threshold
        }


def classify_threat_level(vishing_score: float) -> str:
    """
    Classify threat level based on vishing score.

    Args:
        vishing_score (float): Score between 0 and 1

    Returns:
        str: Threat level classification
    """
    if vishing_score < 0.3:
        return 'LOW'
    elif vishing_score < 0.6:
        return 'MEDIUM'
    elif vishing_score < 0.8:
        return 'HIGH'
    else:
        return 'CRITICAL'


class ComprehensiveVoiceAnalyzer:
    """
    Comprehensive analyzer combining deepfake and vishing detection.
    """

    def __init__(self):
        """Initialize the comprehensive voice analyzer."""
        self.deepfake_model = DeepfakeDetectionModel()
        self.vishing_model = VishingDetectionModel()
        self.auth_model = VoiceAuthenticationModel()

    def analyze_audio(self, audio_path: str, enrolled_speaker: bool = False) -> Dict:
        """
        Perform comprehensive analysis on audio file.

        Args:
            audio_path (str): Path to audio file
            enrolled_speaker (bool): Whether to verify against enrolled speaker

        Returns:
            Dict: Comprehensive analysis results
        """
        deepfake_score, deepfake_analysis = self.deepfake_model.predict_from_audio(
            audio_path
        )
        vishing_score, vishing_analysis = self.vishing_model.predict_from_audio(
            audio_path
        )

        result = {
            'deepfake_detection': deepfake_analysis,
            'vishing_detection': vishing_analysis,
            'overall_risk_score': (deepfake_score + vishing_score) / 2,
            'overall_risk_level': classify_threat_level((deepfake_score + vishing_score) / 2)
        }

        if enrolled_speaker:
            verification = self.auth_model.verify_speaker(audio_path)
            result['speaker_verification'] = verification

        return result

    def train_models(self, X_train: np.ndarray, y_deepfake: np.ndarray,
                     y_vishing: np.ndarray, y_auth: np.ndarray) -> Dict[str, Dict]:
        """
        Train all models simultaneously.

        Args:
            X_train (np.ndarray): Training features
            y_deepfake (np.ndarray): Deepfake labels
            y_vishing (np.ndarray): Vishing labels
            y_auth (np.ndarray): Authentication labels

        Returns:
            Dict[str, Dict]: Training metrics for all models
        """
        metrics = {
            'deepfake_model': self.deepfake_model.train(X_train, y_deepfake),
            'vishing_model': self.vishing_model.train(X_train, y_vishing),
            'auth_model': self.auth_model.train(X_train, y_auth)
        }
        return metrics

    def save_models(self, directory: str) -> None:
        """
        Save all trained models to disk.

        Args:
            directory (str): Directory to save models
        """
        os.makedirs(directory, exist_ok=True)
        
        pickle.dump(self.deepfake_model, open(
            os.path.join(directory, 'deepfake_model.pkl'), 'wb'
        ))
        pickle.dump(self.vishing_model, open(
            os.path.join(directory, 'vishing_model.pkl'), 'wb'
        ))
        pickle.dump(self.auth_model, open(
            os.path.join(directory, 'auth_model.pkl'), 'wb'
        ))

    def load_models(self, directory: str) -> None:
        """
        Load trained models from disk.

        Args:
            directory (str): Directory containing saved models
        """
        self.deepfake_model = pickle.load(open(
            os.path.join(directory, 'deepfake_model.pkl'), 'rb'
        ))
        self.vishing_model = pickle.load(open(
            os.path.join(directory, 'vishing_model.pkl'), 'rb'
        ))
        self.auth_model = pickle.load(open(
            os.path.join(directory, 'auth_model.pkl'), 'rb'
        ))


# Example usage and demonstration
if __name__ == '__main__':
    # Initialize analyzer
    analyzer = ComprehensiveVoiceAnalyzer()

    # Example: Analyze an audio file
    # results = analyzer.analyze_audio('path/to/audio.wav')
    # print(results)

    # To train models with your own data:
    # X_train = ... # Feature matrix
    # y_deepfake = ... # Deepfake labels
    # y_vishing = ... # Vishing labels
    # y_auth = ... # Authentication labels
    # metrics = analyzer.train_models(X_train, y_deepfake, y_vishing, y_auth)
