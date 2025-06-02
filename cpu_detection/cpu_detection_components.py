import torch
import torch.nn as nn
import numpy as np
# import matplotlib.pyplot as plt # Usunięte, bo nie było używane w logice
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler

# enhanced anomaly detection with dedicated features
def extract_advanced_features(data_cpu, seq_len=20, overlap=0.5, scaler=None, fit_scaler=False, users_data=None, time_sin_data=None, time_cos_data=None):
    features = []
    indices = []
    

    step = int(seq_len * (1 - overlap))
    if step == 0: step = 1
    
    for i in range(0, len(data_cpu) - seq_len + 1, step):
        segment_cpu = data_cpu[i:i + seq_len]
        
        # it shows how the cpu is changing over time, it might help to detect sudden anomalies, that is unusual spikes in cpu
        time_features = segment_cpu
        first_derivative = np.diff(segment_cpu, prepend=segment_cpu[0])
        second_derivative = np.diff(first_derivative, prepend=first_derivative[0])
        
        # amplitude features might detect some general energy changes in the cpu signal, but because of randomness might not be very useful
        analytic_signal_cpu = signal.hilbert(segment_cpu)
        amplitude_envelope_cpu = np.abs(analytic_signal_cpu)
        envelope_mean_cpu = np.mean(amplitude_envelope_cpu)
        envelope_std_cpu = np.std(amplitude_envelope_cpu)
        
        # statistical features for cpu, fundamental metrics, show how the cpu is changing and helps extrct anomalies,
        # skewness and kurtosis helps to detect outliers in the cpu usage in a given segment
        mean_cpu = np.mean(segment_cpu)
        std_cpu = np.std(segment_cpu)
        skew_cpu = np.mean(((segment_cpu - mean_cpu) / (std_cpu + 1e-9)) ** 3) if std_cpu > 1e-9 else 0
        kurtosis_cpu = np.mean(((segment_cpu - mean_cpu) / (std_cpu + 1e-9)) ** 4) if std_cpu > 1e-9 else 0
        
        # wavelet (might be useful for detection of sudden and short spikes in cpu usage)
        detail_energy_cpu = 0.0
        if len(segment_cpu) >= 5:
            smoothed_cpu = np.convolve(segment_cpu, np.ones(5)/5, mode='same')
            detail_cpu = segment_cpu - smoothed_cpu
            detail_energy_cpu = np.sum(detail_cpu**2)
        
        feature_vector_cpu = np.concatenate([
            time_features, first_derivative, second_derivative,
            [envelope_mean_cpu, envelope_std_cpu,
            mean_cpu, std_cpu, skew_cpu, kurtosis_cpu, detail_energy_cpu]
        ])

        # contextual features to extract expected patterns
        context_features_list = []
        segment_users = users_data[i:i + seq_len]
        context_features_list.append(np.mean(segment_users))
        segment_time_sin = time_sin_data[i:i + seq_len]
        context_features_list.append(np.mean(segment_time_sin))
        segment_time_cos = time_cos_data[i:i + seq_len]
        context_features_list.append(np.mean(segment_time_cos))

        context_features_array = np.array(context_features_list, dtype=np.float32)
        
        # combine cpu features with contextual features
        feature_vector = np.concatenate([feature_vector_cpu, context_features_array])
        
        features.append(feature_vector)
        indices.append(i)

    features_array = np.array(features, dtype=np.float32)
    
    # the bellow code handles normalization of the features, machine learning models prefer such data to learn better and faster
    if fit_scaler:  # we set fit_scaler to true only for training data, because we want to fit the scaler only once
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
    elif scaler is not None:
        try:
            normalized_features = scaler.transform(features_array)
        except ValueError as e:
            # error might occur if features_array has a different shape than expected by the scaler
            print(f"Error during scaler transformation: {e}")
            print(f"Shape of features_array: {features_array.shape}, expected by scaler features quantity: {scaler.n_features_in_}")
            return np.array([], dtype=np.float32), scaler, np.array(indices, dtype=np.int32)
        
        normalized_features = features_array

    return normalized_features, scaler, np.array(indices, dtype=np.int32)


class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 64
        bottleneck_size = 16
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, bottleneck_size)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, input_size)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class AnomalyDetectorEnsemble:
    pass