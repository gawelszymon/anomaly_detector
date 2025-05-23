import torch
import torch.nn as nn
import numpy as np
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler
from collections import deque

def _extract_raw_features_for_signal_segment(segment):
    # Time domain features
    first_derivative = np.diff(segment)
    second_derivative = np.diff(first_derivative, prepend=first_derivative[0])
    
    # Frequency domain features
    fft_values = np.abs(fft.rfft(segment))
    fft_freqs = fft.rfftfreq(len(segment))
    
    dominant_idx = np.argmax(fft_values)
    dominant_freq = fft_freqs[dominant_idx] if len(fft_freqs) > 0 else 0
    dominant_magnitude = fft_values[dominant_idx] if len(fft_values) > 0 else 0
    
    freq_spread = np.std(fft_values)
    spectral_centroid = np.sum(fft_freqs * fft_values) / (np.sum(fft_values) + 1e-10)
    
    # Phase features
    analytic_signal = signal.hilbert(segment)
    fft_phase = np.angle(fft.rfft(segment))
    phase_mean = np.mean(fft_phase)
    phase_std = np.std(fft_phase)
    
    # Amplitude features
    amplitude_envelope = np.abs(analytic_signal)
    envelope_mean = np.mean(amplitude_envelope)
    envelope_std = np.std(amplitude_envelope)
    
    # Statistical features
    mean = np.mean(segment)
    std = np.std(segment)
    skew = np.mean(((segment - mean) / std) ** 3) if std > 0 else 0
    kurtosis = np.mean(((segment - mean) / std) ** 4) if std > 0 else 0
    
    # Wavelet features
    smoothed = np.convolve(segment, np.ones(5)/5, mode='same')
    detail = segment - smoothed
    detail_energy = np.sum(detail ** 2)
    
    return {
        'segment': segment,
        'first_derivative': first_derivative,
        'second_derivative': second_derivative,
        'dominant_freq': dominant_freq,
        'dominant_magnitude': dominant_magnitude,
        'freq_spread': freq_spread,
        'spectral_centroid': spectral_centroid,
        'phase_mean': phase_mean,
        'phase_std': phase_std,
        'envelope_mean': envelope_mean,
        'envelope_std': envelope_std,
        'mean': mean,
        'std': std,
        'skew': skew,
        'kurtosis': kurtosis,
        'detail_energy': detail_energy
    }
    
def extract_autoencoder_input_vector_from_raw_features(raw_features_dict):
    feature_vector = np.concatenate([
        raw_features_dict['segment'],
        raw_features_dict['first_derivative'],
        raw_features_dict['second_derivative'],
        [raw_features_dict['dominant_freq'], raw_features_dict['dominant_magnitude'],
        raw_features_dict['freq_spread'], raw_features_dict['spectral_centroid'],
        raw_features_dict['phase_mean'], raw_features_dict['phase_std'],
        raw_features_dict['envelope_mean'], raw_features_dict['envelope_std'],
        raw_features_dict['mean'], raw_features_dict['std'], raw_features_dict['skew'],
        raw_features_dict['kurtosis'], raw_features_dict['detail_energy']]
    ])
    return feature_vector.astype(np.float32)
    
def extract_features_for_batch_processing(data, seq_len=20, overlap=0.5):
    features_list = []
    indices = []
    step = int(seq_len * (1 - overlap))
    
    for i in range(0, len(data) - seq_len + 1, step):
        segment = data[i:i + seq_len]
        raw_features_dict = _extract_raw_features_for_signal_segment(segment)
        if raw_features_dict:
            feature_vector = extract_autoencoder_input_vector_from_raw_features(raw_features_dict)
            if feature_vector.size > 0:
                features_list.append(feature_vector)
                indices.append(i)
                
    features_array = np.array(features_list, dtype=np.float32)
    
    scaler = StandardScaler()
    if features_array.shape[0] == 0:
        return np.array([]), scaler, np.array([])
    
    normalized_features = scaler.fit_transform(features_array)
    
    return normalized_features, scaler, np.array(indices)


class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 64
        bottleneck_size = 16
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, bottleneck_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size // 2),
            nn.LeakyReLU(),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, input_size)
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
class AnomalyScorer:
    def __init__(self, recon_error_threshold_params, freq_deviation_threshold_params, phase_deviation_threshold_params, amp_ratio_threshold_params):
        self.re_mean, self.re_std, self.re_sensitivity = recon_error_threshold_params
        self.fd_mean, self.fd_std, self.fd_sensitivity = freq_deviation_threshold_params
        self.pd_mean, self.pd_std, self.pd_sensitivity = phase_deviation_threshold_params
        self.ar_mean, self.ar_std, self.ar_sensitivity = amp_ratio_threshold_params
        
    def check_reconstruction_error(self, noisy_segment_ae_features_scaled, reconstructed_segment_ae_features_scaled):
        mse = np.mean((noisy_segment_ae_features_scaled - reconstructed_segment_ae_features_scaled) ** 2)
        z_score = (mse - self.re_mean) / (self.re_std + 1e-10)
        is_anomaly = z_score > self.re_sensitivity
        return is_anomaly, mse, z_score
    
    def check_frequency_deviation(self, clean_segment_raw_features, noisy_segment_raw_features):
        clean_freq = clean_segment_raw_features.get('dominant_freq', 0)
        noisy_freq = noisy_segment_raw_features.get('dominant_freq', 0)
        freq_diff = np.abs(clean_freq - noisy_freq)
        
        z_score = (freq_diff - self.fd_mean) / (self.fd_std + 1e-10)
        is_anomaly = z_score > self.fd_sensitivity
        return is_anomaly, freq_diff, z_score
    
    def check_phase_deviation(self, clean_segment_raw_features, noisy_segment_raw_features):
        clean_phase_mean = clean_segment_raw_features.get('phase_mean', 0)
        noisy_phase_mean = noisy_segment_raw_features.get('phase_mean', 0)
        phase_diff = np.abs(clean_phase_mean - noisy_phase_mean)

        z_score = (phase_diff - self.pd_mean) / (self.pd_std + 1e-10)
        is_anomaly = z_score > self.pd_sensitivity
        return is_anomaly, phase_diff, z_score

    def check_amplitude_deviation(self, clean_segment_raw_features, noisy_segment_raw_features):
        clean_std = clean_segment_raw_features.get('std', 0)
        noisy_std = noisy_segment_raw_features.get('std', 0)
        
        amp_ratio = clean_std / (noisy_std + 1e-10)
        log_amp_ratio = np.log(amp_ratio + 1e-10)

        z_score = np.abs(log_amp_ratio - self.ar_mean) / (self.ar_std + 1e-10)
        is_anomaly = z_score > self.ar_sensitivity
        return is_anomaly, amp_ratio, z_score
    
    def evaluate_segment(self, noisy_segment_ae_features_scaled, reconstructed_segment_ae_features_scaled, clean_segment_raw_features, noisy_segment_raw_features):
        results = {}
        is_any_anomaly = False
        
        is_re_anomaly, re_score, re_z = self.check_reconstruction_error(noisy_segment_ae_features_scaled, reconstructed_segment_ae_features_scaled)
        results['reconstruction_error'] = {
            'is_anomaly': is_re_anomaly,
            'score': re_score,
            'z_score': re_z
        }
        if is_re_anomaly:
            is_any_anomaly = True
            
        is_fd_anomaly, fd_score, fd_z = self.check_frequency_deviation(clean_segment_raw_features, noisy_segment_raw_features)
        results['frequency_deviation'] = {
            'is_anomaly': is_fd_anomaly, 'score': fd_score, 'z_score': fd_z
        }
        if is_fd_anomaly:
            is_any_anomaly = True
            
        is_pd_anomaly, pd_score, pd_z = self.check_phase_deviation(clean_segment_raw_features, noisy_segment_raw_features)
        results['phase_deviation'] = {
            'is_anomaly': is_pd_anomaly, 'score': pd_score, 'z_score': pd_z
        }
        if is_pd_anomaly:
            is_any_anomaly = True
            
        is_ar_anomaly, ar_score, ar_z = self.check_amplitude_deviation(clean_segment_raw_features, noisy_segment_raw_features)
        results['amplitude_deviation'] = {
            'is_anomaly': is_ar_anomaly, 'score': ar_score, 'z_score': ar_z
        }
        if is_ar_anomaly:
            is_any_anomaly = True
            
        results['is_any_anomaly'] = is_any_anomaly
        return results