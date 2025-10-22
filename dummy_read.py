import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch, cwt, morlet, stft, find_peaks
from scipy.fft import fft, fftshift
from sklearn.ensemble import IsolationForest
from scipy.spatial.distance import mahalanobis

# === CONFIGURATION ===
FEATURE_FIFO = "/home/jfeng/Desktop/Research/Debugging/tee_seer/stm_trust/j/fuzzing_paper/fuzzer/bash_script/feature_fifo"
FFT_SIZE = 2048
SAMPLE_RATE = int(2e6 / 4)  # 400 kHz
ANOMALY_WINDOW_THRESHOLD_IF = 5  # segment flagged if ≥4 anomalous windows (Isolation Forest)
ANOMALY_WINDOW_THRESHOLD_MD = 2  # segment flagged if ≥1 anomalous windows (Mahalanobis)

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# === FEATURE FUNCTIONS ===
def feature_amplitude_mean(signal): return np.mean(np.abs(signal))
def feature_amplitude_std(signal): return np.std(np.abs(signal))
def feature_skew(signal): return skew(np.abs(signal))
def feature_kurtosis(signal): return kurtosis(np.abs(signal), fisher=True)

def feature_correlation(signal, lag=1):
    magnitude = np.abs(signal)
    if lag >= len(magnitude): return np.nan
    corr = np.correlate(magnitude - np.mean(magnitude), magnitude - np.mean(magnitude), mode='full')
    corr = corr[corr.size // 2:]
    corr /= corr[0]
    return corr[lag]

def feature_spectral_entropy(signal, fs, nperseg=None):
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    psd_sum = np.sum(psd)
    if psd_sum == 0: return 0.0
    psd_norm = psd / psd_sum
    return -np.sum(psd_norm * np.log2(psd_norm + 1e-12))

def feature_peakdetection(signal, fs, threshold=0.1):
    N = len(signal)
    spectrum = np.abs(fft(signal))
    spectrum = spectrum[:N // 2]
    spectrum /= np.max(spectrum) if np.max(spectrum) != 0 else 1
    peaks, _ = find_peaks(spectrum, height=threshold)
    return len(peaks)

def feature_spectralcentroid(signal, fs):
    N = len(signal)
    magnitude = np.abs(fft(signal))[:N // 2]
    freqs = np.linspace(0, fs / 2, N // 2)
    denom = np.sum(magnitude)
    return np.sum(freqs * magnitude) / denom if denom != 0 else 0.0

def feature_wavelet(signal):
    scales = np.arange(1, 100)
    coeffs = cwt(signal, morlet, scales)
    magnitudes = np.abs(coeffs)
    total_energy = np.sum(magnitudes**2)
    energy_per_scale = np.sum(magnitudes**2, axis=1)
    energy_sum = np.sum(energy_per_scale)
    weighted_scale = np.sum(energy_per_scale * scales) / energy_sum if energy_sum != 0 else 0.0
    return total_energy, weighted_scale

def feature_stft(signal, fs=1.0, nperseg=256, noverlap=128):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(Zxx)
    total_energy = np.sum(magnitude ** 2)
    energy_per_freq = np.sum(magnitude ** 2, axis=1)
    energy_sum = np.sum(energy_per_freq)
    weighted_freq = np.sum(energy_per_freq * f) / energy_sum if energy_sum != 0 else 0.0
    return total_energy, weighted_freq

def feature_scf(signal, fs, alpha=1000):
    N = len(signal)
    fshift = int(alpha * N / fs)
    spectrum = fftshift(fft(signal))
    if abs(fshift) >= N: return np.nan
    scf_val = np.mean(np.conjugate(spectrum[:N - fshift]) * spectrum[fshift:])
    return np.abs(scf_val)

def feature_caf(signal, fs, alpha=1000, lag=1):
    N = len(signal)
    if lag >= N: return np.nan
    t = np.arange(N - lag) / fs
    term1 = signal[:N - lag]
    term2 = np.conjugate(signal[lag:])
    caf_val = np.mean(term1 * term2 * np.exp(-1j * 2 * np.pi * alpha * t))
    return np.abs(caf_val)

# === SEGMENT PROCESSING ===
def process_iq_segment(segment, segment_index):
    num_windows = len(segment) // FFT_SIZE
    if num_windows == 0:
        return []
    segment = segment[:num_windows * FFT_SIZE]
    windows = segment.reshape(num_windows, FFT_SIZE)
    results = []
    for i, window in enumerate(windows):
        mean_amp = feature_amplitude_mean(window)
        std_amp = feature_amplitude_std(window)
        skew_val = feature_skew(window)
        kurt_val = feature_kurtosis(window)
        corr_val = feature_correlation(window, lag=1)
        spectral_entropy = feature_spectral_entropy(window, SAMPLE_RATE, nperseg=FFT_SIZE)
        peak_count = feature_peakdetection(window, SAMPLE_RATE)
        spec_centroid = feature_spectralcentroid(window, SAMPLE_RATE)
        wavelet_energy, wavelet_scale = feature_wavelet(window)
        stft_energy, stft_freq = feature_stft(window, SAMPLE_RATE)
        scf_val = feature_scf(window, SAMPLE_RATE)
        caf_val = feature_caf(window, SAMPLE_RATE)
        results.append({
            "Segment": f"{segment_index}.{i}",
            "Mean_Amplitude": mean_amp,
            "Std_Amplitude": std_amp,
            "Skew_Val": skew_val,
            "Kurtosis": kurt_val,
            "Autocorrelation": corr_val,
            "Spectral_Entropy": spectral_entropy,
            "Peak_Count": peak_count,
            "Spectral_Centroid": spec_centroid,
            "Wavelet_Total_Energy": wavelet_energy,
            "Wavelet_Weighted_Scale": wavelet_scale,
            "STFT_Total_Energy": stft_energy,
            "STFT_Weighted_Frequency": stft_freq,
            "SCF": scf_val,
            "CAF": caf_val,
        })
    return results

# === LIVE FEATURE EXTRACTION + ANOMALY DETECTION LOOP ===
def live_feature_extractor():
    if not os.path.exists(FEATURE_FIFO):
        print(f"[Error] FIFO {FEATURE_FIFO} does not exist.")
        return

    print(f"[Info] Listening on FIFO: {FEATURE_FIFO}")

    segment_counter = 0
    chunk_size_bytes = FFT_SIZE * 8 * 5

    anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
    training_data = []
    model_trained = False
    mu, cov_inv = None, None  # for Mahalanobis

    total_segments = 0
    total_anomalous_segments = 0

    feature_cols = [
        "Mean_Amplitude", "Std_Amplitude", "Skew_Val", "Kurtosis",
        "Autocorrelation", "Spectral_Entropy", "Peak_Count",
        "Spectral_Centroid", "Wavelet_Total_Energy", "Wavelet_Weighted_Scale",
        "STFT_Total_Energy", "STFT_Weighted_Frequency", "SCF", "CAF"
    ]

    try:
        while True:
            with open(FEATURE_FIFO, "rb") as fifo:
                while True:
                    data_bytes = fifo.read(chunk_size_bytes)
                    if len(data_bytes) == 0:
                        break

                    iq_samples = np.frombuffer(data_bytes, dtype=np.complex64)
                    features = process_iq_segment(iq_samples, segment_counter)
                    segment_counter += 1
                    if not features:
                        continue

                    df = pd.DataFrame(features)
                    for col in ["Wavelet_Total_Energy", "STFT_Total_Energy"]:
                        df[col] = np.log1p(df[col])

                    if not model_trained:
                        training_data.append(df[feature_cols].values)
                        if len(training_data) == 50:
                            all_data = np.vstack(training_data)
                            scaler_mean = np.mean(all_data, axis=0)
                            scaler_std = np.std(all_data, axis=0)
                            scaler_std[scaler_std == 0] = 1
                            all_data_norm = (all_data - scaler_mean) / scaler_std

                            anomaly_detector.fit(all_data_norm)
                            mu = np.mean(all_data_norm, axis=0)
                            cov_inv = np.linalg.pinv(np.cov(all_data_norm, rowvar=False))
                            model_trained = True
                            print("[Info] Anomaly detection models trained.")

                            # --- Print feature importances (Isolation Forest) ---
                            # === Permutation-based feature importance ===
                            print("[Info] Anomaly detection models trained.")

                            # --- Permutation-based feature importance estimation ---
                            print("\n=== Running permutation-based feature importance... ===")

                            # Use the normalized training data for baseline
                            X_test = all_data_norm.copy()
                            clf = anomaly_detector

                            # Compute baseline anomaly scores
                            baseline_scores = clf.decision_function(X_test)

                            importances = []
                            n_repeats = 10  # repeat shuffling for stability

                            for i, feature in enumerate(feature_cols):
                                changes = []
                                for _ in range(n_repeats):
                                    X_permuted = X_test.copy()
                                    np.random.shuffle(X_permuted[:, i])
                                    permuted_scores = clf.decision_function(X_permuted)
                                    # How much do anomaly scores change when this feature is destroyed?
                                    diff = np.mean(np.abs(baseline_scores - permuted_scores))
                                    changes.append(diff)
                                importances.append(np.mean(changes))

                            # --- Build and print importance DataFrame ---
                            imp_df = pd.DataFrame({
                                "Feature": feature_cols,
                                "Importance": importances
                            }).sort_values("Importance", ascending=False)

                            print("\n[Info] Isolation Forest (Permutation-Based) Feature Importances:")
                            print(imp_df.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))

                            print("\n=== Permutation-based Feature Importance ===")
                            print(imp_df)
                        continue

                    # Normalize using global scaler
                    df_norm = (df[feature_cols] - scaler_mean) / scaler_std

                    preds_if = anomaly_detector.predict(df_norm.values)
                    df["IF_Anomaly"] = preds_if

                    scores_md = [mahalanobis(x, mu, cov_inv) for x in df_norm.values]
                    threshold_md = np.mean(scores_md) + 1.5 * np.std(scores_md)
                    df["MD_Anomaly"] = (np.array(scores_md) > threshold_md).astype(int)

                    if_anomaly_count = np.sum(df["IF_Anomaly"] == -1)
                    md_anomaly_count = np.sum(df["MD_Anomaly"] == 1)

                    # Segment-level flag only
                    segment_is_anomalous = (
                        if_anomaly_count >= ANOMALY_WINDOW_THRESHOLD_IF or
                        md_anomaly_count >= ANOMALY_WINDOW_THRESHOLD_MD
                    )

                    total_segments += 1
                    if segment_is_anomalous:
                        total_anomalous_segments += 1
                        print(
                            f"[ALERT] Segment {segment_counter-1} anomalous. "
                            f"IF anomalies: {if_anomaly_count}, MD anomalies: {md_anomaly_count}"
                        )
                    else:
                        print(
                            f"[OK] Segment {segment_counter-1} normal. "
                            f"IF anomalies: {if_anomaly_count}, MD anomalies: {md_anomaly_count}"
                        )

                    # Running stats
                    print(
                        f"[Stats] Total Segments: {total_segments}, "
                        f"Anomalous Segments: {total_anomalous_segments} "
                        f"({100 * total_anomalous_segments / total_segments:.1f}%)"
                    )

    except KeyboardInterrupt:
        print("[Info] Interrupted by user, exiting.")

if __name__ == "__main__":
    live_feature_extractor()
