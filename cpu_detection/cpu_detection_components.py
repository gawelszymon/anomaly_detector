import numpy as np
import torch
import torch.nn as nn
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler


# scaler is a tool to normalize the data, so the data are representing in the same scale.
# This helps the model learn better by not prioritizing one feature over another due to scale differences.
def extract_advanced_features(data_cpu, seq_len=20, overlap=0.5, scaler=None, fit_scaler=False, users_data=None, time_sin_data=None, time_cos_data=None):
    features = []
    indices = []

    step = int(seq_len * (1 - overlap))
    if step == 0: step = 1

    # This loop requires len(data_cpu) to be at least seq_len
    for i in range(0, len(data_cpu) - seq_len + 1, step):
        segment_cpu = data_cpu[i:i + seq_len]

        # Time-domain features: raw signal and its derivatives
        time_features = segment_cpu
        first_derivative = np.diff(segment_cpu, prepend=segment_cpu[0])
        second_derivative = np.diff(first_derivative, prepend=first_derivative[0])

        # Amplitude envelope features from Hilbert transform
        analytic_signal_cpu = signal.hilbert(segment_cpu)
        amplitude_envelope_cpu = np.abs(analytic_signal_cpu)
        envelope_mean_cpu = np.mean(amplitude_envelope_cpu)
        envelope_std_cpu = np.std(amplitude_envelope_cpu)

        # Statistical features
        mean_cpu = np.mean(segment_cpu)
        std_cpu = np.std(segment_cpu)
        skew_cpu = np.mean(((segment_cpu - mean_cpu) / (std_cpu + 1e-9)) ** 3) if std_cpu > 1e-9 else 0
        kurtosis_cpu = np.mean(((segment_cpu - mean_cpu) / (std_cpu + 1e-9)) ** 4) if std_cpu > 1e-9 else 0

        # Simple wavelet-like feature (detail energy)
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

        # Combine CPU features with contextual features
        context_features_list = []
        if users_data is not None and len(users_data) > i + seq_len:
            segment_users = users_data[i:i + seq_len]
            context_features_list.append(np.mean(segment_users))
        if time_sin_data is not None and len(time_sin_data) > i + seq_len:
            segment_time_sin = time_sin_data[i:i + seq_len]
            context_features_list.append(np.mean(segment_time_sin))
        if time_cos_data is not None and len(time_cos_data) > i + seq_len:
            segment_time_cos = time_cos_data[i:i + seq_len]
            context_features_list.append(np.mean(segment_time_cos))

        # Ensure context_features_list has a consistent length
        max_context_features = 3  # Adjust based on the number of optional features
        # Replace zero-padding with the mean of the feature across all segments
        while len(context_features_list) < max_context_features:
            context_features_list.append(np.mean(context_features_list) if context_features_list else 0.0)

        context_features_array = np.array(context_features_list, dtype=np.float32)

        # Combine both sets of features into one final vector
        feature_vector = np.concatenate([feature_vector_cpu, context_features_array])

        features.append(feature_vector)
        indices.append(i)

    features_array = np.array(features, dtype=np.float32)

    # If no features were extracted (because data was too short), return empty structures.
    if features_array.shape[0] == 0:
        return np.array([], dtype=np.float32), None, np.array([], dtype=np.int32)

    # The below code handles normalization of the features
    if fit_scaler:
        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(features_array)
    elif scaler is not None:
        try:
            normalized_features = scaler.transform(features_array)
        except ValueError as e:
            print(f"Error during scaler transformation: {e}")
            print(f"Shape of features_array: {features_array.shape}, expected by scaler features quantity: {scaler.n_features_in_}")
            return np.array([], dtype=np.float32), scaler, np.array(indices, dtype=np.int32)
    else:
        # If no scaler is provided and we are not fitting one, return the unnormalized features.
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
    def __init__(self, X_test_features, reconstructed_features, test_indices, seq_len):
        self.X_test_features = X_test_features
        self.reconstructed_features = reconstructed_features
        self.test_indices = test_indices
        self.seq_len = seq_len
        self.anomalies = set()  # Initialize anomalies as an empty set

    def reconstruction_error_detector(self, sensitivity=2.5):
        if self.X_test_features.shape[0] == 0:
            return np.array([])

        mse = torch.mean((self.X_test_features - self.reconstructed_features) ** 2, dim=1).detach().numpy() # mean squered error for each segment to detect anomaly
        if len(mse) == 0: return np.array([])

        mean_mse = np.mean(mse)
        std_mse = np.std(mse)

        if std_mse < 1e-9: # If the standard deviation is close to zero
            # In such a case, if any error is greater than the mean (or a small tolerance), mark it as an anomaly
            # This handles situations where all errors are almost identical
            anomalies = np.where(mse > mean_mse + 1e-6)[0]
        else:
            z_scores = (mse - mean_mse) / std_mse
            anomalies = np.where(z_scores > sensitivity)[0]

        self.anomalies.update(anomalies)
        return mse

    def _process_signal_segments(self, y_signal, window_size, step, processing_func):   # to simplify te process of processing the signal in a sliding widow
        processed_values = []
        if len(y_signal) < window_size: return np.array([])
        for i in range(0, len(y_signal) - window_size + 1, step):
            segment = y_signal[i:i + window_size]
            processed_values.append(processing_func(segment))
        return np.array(processed_values)

    def frequency_detector(self, y_reference_clean, y_current_noisy, window_size=20, sensitivity=3.0):
        # y_reference_clean: np.array with CPU training data (long signal)
        # y_current_noisy: np.array with currently processed CPU data (test data)

        # We will only use y_current_noisy to find frequency anomalies
        # relative to its own statistics, or comparing to y_reference_clean if it makes sense
        # Simpler: we analyze y_current_noisy for unusual frequencies

        if len(y_current_noisy) < window_size: return np.array([])
        step = window_size // 2
        if step == 0: step = 1

        current_freqs = self._process_signal_segments(
            y_current_noisy, window_size, step,
            lambda seg: np.argmax(np.abs(fft.rfft(seg))) if len(seg) > 0 else 0
        )

        if len(current_freqs) < 2 : return np.array([]) # We need at least 2 values to calculate z_score

        # Simple detection: deviations from the average frequency in y_current_noisy
        freq_z_scores = (current_freqs - np.mean(current_freqs)) / (np.std(current_freqs) + 1e-10)
        freq_anomalies_indices_in_current_freqs = np.where(np.abs(freq_z_scores) > sensitivity)[0] # abs for deviations in both directions

        # Map to the original indices of test_indices
        for anom_idx_in_freqs in freq_anomalies_indices_in_current_freqs:
            # position in y_current_noisy
            position_in_y_noisy = anom_idx_in_freqs * step
            # Find the closest segment index in self.test_indices
            # This mapping is approximate because test_indices refers to segments for the autoencoder,
            # and window_size and step here may be different.
            # For simplicity, we map based on the position in the original signal.
            if len(self.test_indices) > 0: # test_indices is for reference to segments in a signal, so we can tag which segment contains an anomaly
                closest_segment_idx = np.argmin(np.abs(self.test_indices - position_in_y_noisy))
                self.anomalies.add(closest_segment_idx)
        return current_freqs # Return the calculated frequencies for possible analysis


    def phase_detector(self, y_reference_clean, y_current_noisy, window_size=20, sensitivity=2.0):
        if len(y_current_noisy) < window_size: return np.array([])
        step = window_size // 2
        if step == 0: step = 1

        current_phases_rate = self._process_signal_segments(
            y_current_noisy, window_size, step,
            lambda seg: np.mean(np.diff(np.unwrap(np.angle(signal.hilbert(seg))))) if len(seg) > 1 else 0
        )

        if len(current_phases_rate) < 2 : return np.array([])

        phase_z_scores = (current_phases_rate - np.mean(current_phases_rate)) / (np.std(current_phases_rate) + 1e-10)
        phase_anomalies_indices = np.where(np.abs(phase_z_scores) > sensitivity)[0]

        for anom_idx_in_phases in phase_anomalies_indices:
            position_in_y_noisy = anom_idx_in_phases * step
            if len(self.test_indices) > 0:
                closest_segment_idx = np.argmin(np.abs(self.test_indices - position_in_y_noisy))
                self.anomalies.add(closest_segment_idx)
        return current_phases_rate

    def amplitude_detector(self, y_reference_clean, y_current_noisy, window_size=20, sensitivity=2.0):
        if len(y_current_noisy) < window_size: return np.array([])
        step = window_size // 2
        if step == 0: step = 1

        current_amps_std = self._process_signal_segments(
            y_current_noisy, window_size, step,
            lambda seg: np.std(seg) if len(seg) > 0 else 0
        )

        if len(current_amps_std) < 2 : return np.array([])

        amp_z_scores = (current_amps_std - np.mean(current_amps_std)) / (np.std(current_amps_std) + 1e-10)
        amp_anomalies_indices = np.where(np.abs(amp_z_scores) > sensitivity)[0]

        for anom_idx_in_amps in amp_anomalies_indices:
            position_in_y_noisy = anom_idx_in_amps * step
            if len(self.test_indices) > 0:
                closest_segment_idx = np.argmin(np.abs(self.test_indices - position_in_y_noisy))
                self.anomalies.add(closest_segment_idx)
        return current_amps_std

    def run_all_detectors(self, y_reference_clean_signal, y_current_noisy_signal):

        # y_reference_clean_signal: np.array with CPU training data (long signal)
        # y_current_noisy_signal: np.array with currently processed CPU data (test data)

        mse = self.reconstruction_error_detector()
        freq_metric = self.frequency_detector(y_reference_clean_signal, y_current_noisy_signal)
        phase_metric = self.phase_detector(y_reference_clean_signal, y_current_noisy_signal)
        amp_metric = self.amplitude_detector(y_reference_clean_signal, y_current_noisy_signal)

        # Add descriptive statistics for each metric
        metrics_summary = {
            'reconstruction_error': {
                'mean': np.mean(mse) if mse.size > 0 else None,
                'std': np.std(mse) if mse.size > 0 else None,
                'values': mse
            },
            'frequency_metric': {
                'mean': np.mean(freq_metric) if freq_metric.size > 0 else None,
                'std': np.std(freq_metric) if freq_metric.size > 0 else None,
                'values': freq_metric
            },
            'phase_metric': {
                'mean': np.mean(phase_metric) if phase_metric.size > 0 else None,
                'std': np.std(phase_metric) if phase_metric.size > 0 else None,
                'values': phase_metric
            },
            'amplitude_metric': {
                'mean': np.mean(amp_metric) if amp_metric.size > 0 else None,
                'std': np.std(amp_metric) if amp_metric.size > 0 else None,
                'values': amp_metric
            }
        }

        print(f"Metrics summary: {metrics_summary}")
        return list(self.anomalies), metrics_summary
