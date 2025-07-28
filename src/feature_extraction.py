import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # 13 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs, axis=1)

    # 1 ZCR
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=y))

    # 1 RMSE
    rmse = np.mean(librosa.feature.rms(y=y))

    # 1 Spectral Centroid
    spec_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))

    # 1 Spectral Rolloff
    spec_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

    # 1 Spectral Bandwidth ✅ (This is the missing one!)
    spec_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))

    # 1 Chroma STFT
    chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))

    # 1 Spectral Contrast
    contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))

    # 1 Tonnetz
    try:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr))
    except:
        tonnetz = 0.0

    # 1 Tempo
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    # Final: 13 + 9 = 22 features
    features = np.array([
        *mfccs_mean,
        zcr, rmse, spec_centroid, spec_rolloff,
        spec_bandwidth, chroma, contrast, tonnetz, tempo
    ])

    return features
