import numpy as np
import glob
import os
import pandas as pd
from scipy.stats import skew
from scipy.signal import welch
from scipy.signal import cwt, morlet
from tftb.processing import WignerVilleDistribution
from scipy.signal import stft
from scipy.signal import decimate
from scipy.stats import kurtosis
from scipy.signal import find_peaks
from numpy.fft import fft, fftshift, fftfreq

# Loads iq file from a given filepath
def load_iq_file(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    if len(raw) % 2 != 0:
        raise ValueError(f"Invalid IQ file")
    iq_data = raw[::2] + 1j * raw[1::2]
    return iq_data

def decimate_iq(iq_samples, D, num_taps=101):
    return decimate(iq_samples, D, ftype='fir', zero_phase=False)

### TIME SERIES ###

# Mean of time series amplitude
def feature_amplitude_mean(signal):
    magnitude = abs(signal)
    return np.mean(magnitude)

# Standard Deviation of time series amplitude
def feature_amplitude_std(signal):
    magnitude = abs(signal)
    return np.std(magnitude)

# Skew of signal
def feature_skew(signal):
    magnitude = abs(signal)
    return skew(magnitude)

# Kurtosis (tailedness of amplitude distribution)
def feature_kurtosis(signal):
    magnitude = np.abs(signal)
    return kurtosis(magnitude, fisher=True)  # low kurtosis means noisy (flat)

# Correlation
def feature_correlation(signal, lag=1):
    magnitude = np.abs(signal)
    if lag >= len(magnitude): # check valid lag
        return np.nan

    corr = np.correlate(magnitude - np.mean(magnitude),
                        magnitude - np.mean(magnitude), mode='full') # cross-correlation of signal with itself
    corr = corr[corr.size // 2:]  # only keep non-negative lags
    corr /= corr[0]  # normalize s.t. corr[0] = 1
    return corr[lag]


###################################

### Frequency Analysis ###
def feature_spectral_entropy(signal, fs, nperseg=None):
    # Estimate power spectral density (PSD) using Welch's method
    freqs, psd = welch(signal, fs, nperseg=nperseg)
    
    # Normalize the PSD
    psd_norm = psd / np.sum(psd)
    
    # Compute spectral entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))  # small epsilon to avoid log(0)
    return entropy

# Peak Detection
def feature_peakdetection(signal, fs, threshold=0.1):
    N = len(signal)
    spectrum = np.abs(fft(signal)) # get fft of signal
    spectrum = spectrum[:N // 2] # keep positive frequencies
    spectrum /= np.max(spectrum)  # normalize

    peaks, _ = find_peaks(spectrum, height=threshold)
    return len(peaks)

# Central Spectroid
def feature_spectralcentroid(signal, fs):
    N = len(signal)
    magnitude = np.abs(fft(signal))[:N // 2] # take fft and only keep positive frequencies
    freqs = np.linspace(0, fs / 2, N // 2) # keep nyquist num of samples
    centroid = np.sum(freqs * magnitude) / np.sum(magnitude) # weighted mean frequency
    return centroid

###################################

### Time + Frequency Analysis ###
# Wavelet Transform

def feature_wavelet(signal):
    # Decompose real and imaginary parts separately
    # set the wavelet scales
    scales = np.arange(1, 100)
    coeffs = cwt(signal, morlet, scales)
    magnitudes = np.abs(coeffs)

    # Total energy
    total_energy = np.sum(magnitudes**2)

    # Mean scale weighted by energy (a proxy for dominant frequency)
    energy_per_scale = np.sum(magnitudes**2, axis=1)
    weighted_scale = np.sum(energy_per_scale * scales) / np.sum(energy_per_scale)

    return np.array([total_energy, weighted_scale])

# Wigner Ville
def feature_wigner_ville(signal):
    # Compute Wigner-Ville Distribution
    tfr = WignerVilleDistribution(signal)
    tfr.run()
    wvd_matrix = tfr.tfr  # shape: (n_freqs, n_times)

    wvd_abs = np.abs(wvd_matrix)

    # Total energy
    total_energy = np.sum(wvd_abs ** 2)

    # Energy per frequency (sum over time)
    energy_per_freq = np.sum(wvd_abs ** 2, axis=1)
    freqs = np.arange(wvd_matrix.shape[0])

    # Weighted frequency (proxy for dominant frequency)
    weighted_freq = np.sum(energy_per_freq * freqs) / np.sum(energy_per_freq)

    return np.array([total_energy, weighted_freq])

# Short Time Fourier Transform
def feature_stft(signal, fs=1.0, nperseg=256, noverlap=128):
    # Compute STFT
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, noverlap=noverlap)

    magnitude = np.abs(Zxx)

    # Total energy in the spectrogram
    total_energy = np.sum(magnitude ** 2)

    # Energy per frequency
    energy_per_freq = np.sum(magnitude ** 2, axis=1)

    # Weighted average frequency (proxy for dominant frequency)
    weighted_freq = np.sum(energy_per_freq * f) / np.sum(energy_per_freq)

    return np.array([total_energy, weighted_freq])

# scf
def feature_scf(signal, fs, alpha=0.1):
    N = len(signal)
    fshift = int(alpha * N / fs) # shift index based on cyclic freq
    spectrum = fftshift(fft(signal))
    if abs(fshift) >= N: # skip invalid shift values
        return np.nan
    scf_val = np.mean(np.conjugate(spectrum[:N - fshift]) * spectrum[fshift:]) # cyclic cross correlation
    return np.abs(scf_val)

# caf
def feature_caf(signal, fs, alpha=0.1, lag=10):
    N = len(signal)
    if lag >= N: # check that lag is valid
        return np.nan
    t = np.arange(N - lag) / fs
    term1 = signal[:N - lag]
    term2 = np.conjugate(signal[lag:])
    caf_val = np.mean(term1 * term2 * np.exp(-1j * 2 * np.pi * alpha * t)) # the equation 
    return np.abs(caf_val)

###################################

# Read iq files from input folder
input_folder = "/Users/amy/Documents/GitHub/sc_fuzzing/data"
# Parameters
samp_rate = 2e6
fcent = 2.402e9
fftsize = 8912
downsample_factor = 5
samp_rate = int(samp_rate / downsample_factor)

iq_files = sorted(glob.glob(os.path.join(input_folder, "*.iq")))

# Iterate through all files
for file in iq_files:
    results = []
    iq_data = load_iq_file(file)
    
    # Downsample if desired
    iq_data = decimate_iq(iq_data, downsample_factor)

    # Iterate over windows
    num_segments = len(iq_data) // fftsize
    iq_data = iq_data[:num_segments * fftsize] 
    iq_data = iq_data.reshape(num_segments, fftsize)

    # Counter to track progres
    counter = 0

    for seg in iq_data:
        # Extract features
        
        # Time Analysis
        # Mean of Amplitude
        mean_amplitude = feature_amplitude_mean(seg)
        # Standard Deviation of Amplitude
        mean_std = feature_amplitude_std(seg)
        # TODO: Skew
        skew_val = feature_skew(seg)
        # TODO: Autocorrelation. Returns the autocorrelation for a chosen lag
        autocorrelation_val = feature_correlation(seg)
        # TODO: Kurtosis. Returns the kurtosis of this segment
        kurtosis_val = feature_kurtosis(seg)
        # Frequency Analysis
        # TODO: Num Peaks. Return the number of peaks seen in this segment (for FFT)
        num_peaks = feature_peakdetection(seg, samp_rate)
        # TODO: Spectral Centroid. Returns the frequency of the center of the energy in this segment (for FFT)
        spectral_centroid = feature_spectralcentroid(seg, samp_rate)
        # Spectral Entropy
        spectral_entropy = feature_spectral_entropy(seg, samp_rate, fftsize)

        # Time + Frequency Analysis
        # Wavelet Transform
        wavelet_energy, wavelet_weighted_scale = feature_wavelet(seg)
        # Wigner Ville
        wvd_energy, wvd_weighted_freq = feature_wigner_ville(seg)
        # STFT
        stft_energy, stft_weighted_freq = feature_stft(seg, samp_rate)
        # TODO: SCF or CAF. Not sure what number(s) we should return yet
        scf_val = feature_scf(seg, samp_rate, alpha=1e5)
        caf_val = feature_caf(seg, samp_rate, alpha=1e5, lag=10)
        # Store results
        results.append({
            "Filename": os.path.basename(file),
            "Segment": counter,
            "Mean_Amplitude": mean_amplitude,
            "Std_Amplitude": mean_std,
            "Skew_Val": skew_val,
            "Kurtosis": kurtosis_val,
            "Autocorrelation": autocorrelation_val,
            "Spectral_Entropy": spectral_entropy,
            "Num_Peaks": num_peaks,
            "Spectral_Centroid": spectral_centroid,
            "Wavelet_Total_Energy": wavelet_energy,
            "Wavelet_Weighted_Scale": wavelet_weighted_scale,
            "STFT_Total_Energy": stft_energy,
            "STFT_Weighted_Frequency": stft_weighted_freq,
            "SCF_Val": scf_val,
            "CAF_Val": caf_val,
        })
        
           
        counter = counter + 1
        print("Progress: ", counter, "/", num_segments)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Add all numeric features here
    feature_cols = [
        "Mean_Amplitude",
        "Std_Amplitude",
        "Skew_Val",
        "Kurtosis",
        "Autocorrelation",
        "Spectral_Entropy",
        "Num_Peaks",
        "Spectral_Centroid",
        "Wavelet_Total_Energy",
        "Wavelet_Weighted_Scale",
        "STFT_Total_Energy",
        "STFT_Weighted_Frequency",
        "SCF_Val",
        "CAF_Val"
    ]
    
    # Optional: log-transform energy features to reduce dynamic range
    df["Wavelet_Total_Energy"] = np.log1p(df["Wavelet_Total_Energy"])
    df["STFT_Total_Energy"] = np.log1p(df["STFT_Total_Energy"])

    # Standardize features: zero mean, unit variance
    df[feature_cols] = (df[feature_cols] - df[feature_cols].mean()) / df[feature_cols].std()

    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(file))[0]

    # Save CSV and Excel named per file
    csv_path = f"{base_name}_features.csv"
    xlsx_path = f"{base_name}_features.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"Features saved for {base_name}: {csv_path}, {xlsx_path}")
