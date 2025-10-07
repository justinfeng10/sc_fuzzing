import numpy as np
import glob
import os
import pandas as pd
from scipy.signal import cwt, morlet
from tftb.processing import WignerVilleDistribution
from scipy.signal import stft

# Loads iq file from a given filepath
def load_iq_file(filepath):
    raw = np.fromfile(filepath, dtype=np.float32)
    if len(raw) % 2 != 0:
        raise ValueError(f"Invalid IQ file")
    iq_data = raw[::2] + 1j * raw[1::2]
    return iq_data

def downsample(signal, factor):
    return signal[::factor]


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
    iq_data = downsample(iq_data, downsample_factor)

    # Iterate over windows
    num_segments = len(iq_data) // fftsize
    iq_data = iq_data[:num_segments * fftsize] 
    iq_data = iq_data.reshape(num_segments, fftsize)

    # Counter to track progres
    counter = 0

    for seg in iq_data:
        # Extract features
        
        # Time + Frequency Analysis
        # Wavelet Transform
        wavelet_energy, wavelet_weighted_scale = feature_wavelet(seg)
        # Wigner Ville
        
        #wvd_energy, wvd_weighted_freq = feature_wigner_ville(seg)
        
        
        # New
        stft_energy, stft_weighted_freq = feature_stft(seg, samp_rate)
        
        # Store results
        results.append({
            "Filename": os.path.basename(file),
            "Segment": counter,
            "Wavelet_Total_Energy": wavelet_energy,
            "Wavelet_Weighted_Scale": wavelet_weighted_scale,
            "STFT_Total_Energy": stft_energy,
            "STFT_Weighted_Frequency": stft_weighted_freq,
        })
        
           
        counter = counter + 1
        print("Progress: ", counter, "/", num_segments)

    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    feature_cols = ["Wavelet_Total_Energy", "Wavelet_Weighted_Scale", "STFT_Total_Energy", "STFT_Weighted_Frequency"]
    
    # Optional: log-transform energy features to reduce dynamic range
    df["Wavelet_Total_Energy"] = np.log1p(df["Wavelet_Total_Energy"])
    df["STFT_Total_Energy"] = np.log1p(df["STFT_Total_Energy"])

    df[feature_cols] = (df[feature_cols] - df[feature_cols].min()) / (df[feature_cols].max() - df[feature_cols].min())


    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(file))[0]

    # Save CSV and Excel named per file
    csv_path = f"{base_name}_features.csv"
    xlsx_path = f"{base_name}_features.xlsx"

    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)

    print(f"Features saved for {base_name}: {csv_path}, {xlsx_path}")
