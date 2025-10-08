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

###################################

# Read iq files from input folder
input_folder = "/home/jfeng/Desktop/Research/Debugging/tee_seer/stm_trust/j/fuzzing_paper/fuzzer/bash_script/snipuzz_teeseer"
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
        # TODO: Kurtosis. Returns the kurtosis of this segment

        # Frequency Analysis
        # TODO: Num Peaks. Return the number of peaks seen in this segment (for FFT)
        # TODO: Spectral Centroid. Returns the frequency of the center of the energy in this segment (for FFT)
        # Spectral Entropy
        spectral_entropy = feature_spectral_entropy(seg, samp_rate, fftsize)

        # Time + Frequency Analysis
        # Wavelet Transform
        wavelet_energy, wavelet_weighted_scale = feature_wavelet(seg)
        # Wigner Ville
        #wvd_energy, wvd_weighted_freq = feature_wigner_ville(seg)
        # STFT
        stft_energy, stft_weighted_freq = feature_stft(seg, samp_rate)
        # TODO: SCF or CAF. Not sure what number(s) we should return yet
        
        # Store results
        results.append({
            "Filename": os.path.basename(file),
            "Segment": counter,
            "Mean_Amplitude": mean_amplitude,
            "Std_Amplitude": mean_std,
            "Skew_Val": skew_val,
            "Spectral_Entropy": spectral_entropy,
            "Wavelet_Total_Energy": wavelet_energy,
            "Wavelet_Weighted_Scale": wavelet_weighted_scale,
            "STFT_Total_Energy": stft_energy,
            "STFT_Weighted_Frequency": stft_weighted_freq,
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
        "Spectral_Entropy",
        "Wavelet_Total_Energy",
        "Wavelet_Weighted_Scale",
        "STFT_Total_Energy",
        "STFT_Weighted_Frequency"
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
