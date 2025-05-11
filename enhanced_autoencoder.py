import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler

x = np.linspace(0, 10 * np.pi, 1000)
y_clean = np.abs(np.sin(x)).astype(np.float32)
y_noisy = y_clean.copy()

# common anomalies
y_noisy[12] -= 0.8
y_noisy[444] += 0.7
y_noisy[111] += 1.0
y_noisy[100:110] += 0.5

# silient anomalies
x_3x = x[700:720]
y_noisy[700:720] = np.abs(np.sin(3 * x_3x)).astype(np.float32)  # 3x frequency
x_plus_phase = x[800:820]
y_noisy[800:820] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)  # phase shift
x_attenuator = x[900:920]
y_noisy[900:920] = np.abs(np.sin(x_attenuator) * 0.1).astype(np.float32)  # attenuator

# enhaced anomaly detection with dedicated features
def extract_advanced_features(data, seq_len=20, overlap=0.5):
    features = []
    indices = []
    step = int(seq_len * (1 - overlap))
    
    for i in range(0, len(data) - seq_len + 1, step):
        segment = data[i:i + seq_len]
        
        # TIME DOMAIN FEATUERES
        time_features = segment
        
        # rate of change in the time domain (analyzing how signal changes in a time)
        first_derivative = np.diff(segment, prepend=segment[0])
        second_derivative = np.diff(first_derivative, prepend=first_derivative[0])
        
        # FREQUENCY DOMAIN FEATURES
        fft_values = np.abs(fft.rfft(segment)) # fft fast fourier transform we change the signal from time domain into frequency domain, so we can detect frequency anomalies
        fft_freqs = fft.rfftfreq(len(segment))
        
        # obtaining the dominant frequency and its magnitude
        dominant_idx = np.argmax(fft_values)
        dominant_freq = fft_freqs[dominant_idx]
        dominant_magnitude = fft_values[dominant_idx]
        
        # frequency spread - the measure of energy distribution across frequency bands to help detect when a signal shifts from one frequency to another
        # the rise of the frequency spread indicates that the signal is changing the frequency so it might be an anomaly
        freq_spread = np.std(fft_values)
        
        # spectral centroid - weighted mean of frequencies (detects frequency shifts)
        spectral_centroid = np.sum(fft_freqs * fft_values) / np.sum(fft_values) if np.sum(fft_values) > 0 else 0
        
        # PHASE FEATURES
        # compute phase of FFT components - helps detect phase shifts
        fft_phase = np.angle(fft.rfft(segment))
        phase_mean = np.mean(fft_phase) # mean phase shift
        phase_std = np.std(fft_phase) #it show how the phase of certain segment is changing
        
        # AMPLITUDE FEATURES
        # signal envelope (obwiednia) using Hilbert's transform (detects amplitude modulation)
        analytic_signal = signal.hilbert(segment) # Hilbert transform - add the imaginary part to the signal
        amplitude_envelope = np.abs(analytic_signal) # envelope of the signal lets us see how the amplitude of the signal changes over time
        envelope_mean = np.mean(amplitude_envelope) # tell us how the the signal is strong in such a segment
        envelope_std = np.std(amplitude_envelope)   # how the amplitude of the signal is changing in such a segment
        
        # statistical features of our signal's segment
        mean = np.mean(segment) # tell us if the signal is moved up or down
        std = np.std(segment) # standard deviation - how the signal is changing, the high value tells us that the signal is changing a lot
        skew = np.mean(((segment - mean) / std) ** 3) if std > 0 else 0 # skewness (skosność) - tell us if the signal is symmetric or not, 0 means the gaussian distribution
        kurtosis = np.mean(((segment - mean) / std) ** 4) if std > 0 else 0 # measure the taildness of the distribution, high value means that the signal has a lot of outliers
        
        # Wavelet for multiple resolutions analysis, analyzing the signal at different scales
        smoothed = np.convolve(segment, np.ones(5)/5, mode='same')  # it takes the next 5 values of the signal and takes the mean of them
        #convoleve is a splot function mave the filter over the signal and for each position it takes the mean of each part, mode='same' means the outout same size as input
        detail = segment - smoothed # extract sharp details and anomalies from the original signal to find the anomalies
        detail_energy = np.sum(detail**2)
        
        # feature combination
        # we combine all the features into a single vector, that is prepared for the autoencoder
        feature_vector = np.concatenate([
            time_features,                      # raw signal values
            first_derivative,                   # first derivative to betray the rate of change
            second_derivative,                  # the second derivative to betray the acceleration of the signal changes
            [dominant_freq, dominant_magnitude,
            freq_spread, spectral_centroid,
            phase_mean, phase_std,
            envelope_mean, envelope_std,
            mean, std, skew, kurtosis,
            detail_energy]
        ])
        
        features.append(feature_vector)
        indices.append(i)
    
    features_array = np.array(features, dtype=np.float32)
    
    scaler = StandardScaler() # the normalization object, works for the features by substracting the mean and dividing by the standard deviation
    normalized_features = scaler.fit_transform(features_array) # fit for mean and std of the training data
    
    return normalized_features, scaler, np.array(indices)   # normalized_features - prepared date for the autoencoder, scaler - the object to  normalize and scale the date for later use, np.array(indices) - the beginning index of each segemnt

# improved autoencoder with more layers and dropout to capture more complex and less linear relationships
# input -> hidden -> hidden/2 -> bottleneck -> hidden/2 -> hidden -> output
class AdvancedAutoencoder(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 64    # larger hidden layer to capture more complex and in genral more relationships, the space to learn some features before the bottleneck
        bottleneck_size = 16    # narrower bottleneck to force the model to learn only the most important features
        
        # overfitting the situation, when the model learns the trainging date too well and it does not generalize to the new data, so it cannot extract the most important feature
        # to prevent overfitting we use dropout, which randomly sets some neurons to 0 during training, so the model cannot rely on any specific neuron and learn the general context
        
        # enhanced encoder takes input and compresses it into a lower-dimensional representation (latent vector)
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2),  # it does not cut the negative values, it is not so strong as ReLU
            nn.Dropout(0.2),  # Prevent overfitting by randomly setting some neurons to 0
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, bottleneck_size)
        )
        
        # enhanced decoder takes the latent vector and reconstructs the original input
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

# Multi-detector approach - different methods for different anomalies, it lets us to detect different types of anomalies
class AnomalyDetectorEnsemble:
    def __init__(self, X_test, reconstructed, test_indices, seq_len):
        self.X_test = X_test
        self.reconstructed = reconstructed
        self.test_indices = test_indices
        self.seq_len = seq_len
        self.anomalies = set()
        
    # for basic reconstruction error detector
    def reconstruction_error_detector(self, sensitivity=2.5):
        mse = torch.mean((self.X_test - self.reconstructed) ** 2, dim=1).numpy()
        z_scores = (mse - np.mean(mse)) / np.std(mse)
        anomalies = np.where(z_scores > sensitivity)[0]
        self.anomalies.update(anomalies)
        return mse
        
    def frequency_detector(self, y_clean, y_noisy, window_size=20, sensitivity=3.0):
        clean_freqs = []
        noisy_freqs = []
        
        for i in range(0, len(y_clean) - window_size + 1, window_size // 2):
            # to get the frequency of the clean signal
            clean_segment = y_clean[i:i + window_size]
            clean_fft = np.abs(fft.rfft(clean_segment))
            clean_freqs.append(np.argmax(clean_fft))
            
            # get the frequency of noisy signal
            noisy_segment = y_noisy[i:i + window_size]
            noisy_fft = np.abs(fft.rfft(noisy_segment))
            noisy_freqs.append(np.argmax(noisy_fft))
        
        # calculate frequency difference
        freq_diff = np.abs(np.array(clean_freqs) - np.array(noisy_freqs))
        freq_z_scores = (freq_diff - np.mean(freq_diff)) / (np.std(freq_diff) + 1e-10)
        
        # find and mark anomalies
        freq_anomalies = np.where(freq_z_scores > sensitivity)[0]
        
        # map to test indices
        for anomaly_idx in freq_anomalies:
            position = anomaly_idx * (window_size // 2)
            closest_idx = np.argmin(np.abs(self.test_indices - position))
            self.anomalies.add(closest_idx)
            
        return freq_diff
    
    def phase_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0):
        clean_phases = []
        noisy_phases = []
        
        for i in range(0, len(y_clean) - window_size + 1, window_size // 2):
            # clean signal phase
            clean_segment = y_clean[i:i + window_size]
            clean_analytic = signal.hilbert(clean_segment)
            clean_phase = np.unwrap(np.angle(clean_analytic))
            clean_phases.append(np.mean(np.diff(clean_phase)))
            
            # noisy signal phase
            noisy_segment = y_noisy[i:i + window_size]
            noisy_analytic = signal.hilbert(noisy_segment)
            noisy_phase = np.unwrap(np.angle(noisy_analytic))
            noisy_phases.append(np.mean(np.diff(noisy_phase)))
        
        # Phase difference indicator
        phase_diff = np.abs(np.array(clean_phases) - np.array(noisy_phases))
        phase_z_scores = (phase_diff - np.mean(phase_diff)) / (np.std(phase_diff) + 1e-10)
        
        phase_anomalies = np.where(phase_z_scores > sensitivity)[0]
        
        # to test indices
        for anomaly_idx in phase_anomalies:
            position = anomaly_idx * (window_size // 2)
            closest_idx = np.argmin(np.abs(self.test_indices - position))
            self.anomalies.add(closest_idx)
            
        return phase_diff
    
    def amplitude_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0):
        clean_amps = []
        noisy_amps = []
        
        for i in range(0, len(y_clean) - window_size + 1, window_size // 2):
            clean_segment = y_clean[i:i + window_size]
            clean_amps.append(np.std(clean_segment))
            
            noisy_segment = y_noisy[i:i + window_size]
            noisy_amps.append(np.std(noisy_segment))
        
        # Amplitude ratio (robust to scale differences)
        amp_ratio = np.array(clean_amps) / (np.array(noisy_amps) + 1e-10)
        log_amp_ratio = np.log(amp_ratio + 1e-10)
        amp_z_scores = np.abs(log_amp_ratio - np.mean(log_amp_ratio)) / (np.std(log_amp_ratio) + 1e-10)
        
        amp_anomalies = np.where(amp_z_scores > sensitivity)[0]
        # Map to test indices
        for anomaly_idx in amp_anomalies:
            position = anomaly_idx * (window_size // 2)
            closest_idx = np.argmin(np.abs(self.test_indices - position))
            self.anomalies.add(closest_idx)
            
        return amp_ratio
    
    def run_all_detectors(self, y_clean, y_noisy):
        mse = self.reconstruction_error_detector()
        freq_diff = self.frequency_detector(y_clean, y_noisy)
        phase_diff = self.phase_detector(y_clean, y_noisy)
        amp_ratio = self.amplitude_detector(y_clean, y_noisy)
        
        return list(self.anomalies), {
            'reconstruction_error': mse,
            'frequency_difference': freq_diff,
            'phase_difference': phase_diff,
            'amplitude_ratio': amp_ratio
        }

# Main execution
def main():
    # Extract enhanced features with overlap
    seq_len = 20
    X_train_features, train_scaler, train_indices = extract_advanced_features(y_clean, seq_len, overlap=0.75)
    X_test_features, _, test_indices = extract_advanced_features(y_noisy, seq_len, overlap=0.75)
    
    # Apply training scaler to test features
    X_test_features = train_scaler.transform(X_test_features)
    
    # Prepare tensors
    X_train = torch.tensor(X_train_features, dtype=torch.float32)
    X_test = torch.tensor(X_test_features, dtype=torch.float32)
    
    # Create and train model
    input_size = X_train.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training with gradient clipping and early stopping
    epochs = 10000
    patience = 100
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Forward pass
        output = model(X_train)
        loss = criterion(output, X_train)
        
        # Backward pass with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        
        # Early stopping logic
        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #     if patience_counter >= patience:
        #         print(f"Early stopping at epoch {epoch}")
        #         break
                
        # Additional early stopping based on absolute loss value
        if loss.item() < 0.0001:
            print(f"Loss threshold reached at epoch {epoch}")
            break
    
    # Evaluate model and detect anomalies
    with torch.no_grad():
        reconstructed = model(X_test)
    
    # Use ensemble detector for better detection of different anomaly types
    ensemble = AnomalyDetectorEnsemble(X_test, reconstructed, test_indices, seq_len)
    anomaly_indices, metrics = ensemble.run_all_detectors(y_clean, y_noisy)
    
    # Visualize results with multiple plots
    plt.figure(figsize=(15, 15))
    
    # Plot 1: Original signal with detected anomalies
    plt.subplot(4, 1, 1)
    plt.plot(y_clean, 'g-', alpha=0.5, label='Clean signal')
    plt.plot(y_noisy, 'b-', label='Noisy signal')
    
    # Highlight detected anomalies
    for idx in anomaly_indices:
        start_idx = test_indices[idx]
        plt.axvspan(start_idx, start_idx + seq_len, color='red', alpha=0.3)
    
    # Highlight true anomaly regions
    plt.axvspan(700, 720, color='yellow', alpha=0.2, label='Frequency anomaly (3x)')
    plt.axvspan(800, 820, color='orange', alpha=0.2, label='Phase shift anomaly')
    plt.axvspan(900, 920, color='purple', alpha=0.2, label='Amplitude anomaly')
    plt.legend()
    plt.title('Signal with Multi-Detector Anomalies')
    
    # Plot 2: Reconstruction error
    plt.subplot(4, 1, 2)
    plt.plot(test_indices, metrics['reconstruction_error'], 'b-')
    threshold = np.mean(metrics['reconstruction_error']) + 2.5 * np.std(metrics['reconstruction_error'])
    plt.axhline(y=threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Position in signal')
    plt.ylabel('Reconstruction error')
    plt.title('Reconstruction Error Analysis')
    
    # Plot 3: Frequency analysis
    plt.subplot(4, 1, 3)
    plt.plot(metrics['frequency_difference'], 'g-')
    freq_threshold = np.mean(metrics['frequency_difference']) + 3.0 * np.std(metrics['frequency_difference'])
    plt.axhline(y=freq_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Window index')
    plt.ylabel('Frequency difference')
    plt.title('Frequency Analysis (Detects 3x Frequency Change)')
    
    # Plot 4: Amplitude ratio analysis
    plt.subplot(4, 1, 4)
    plt.plot(metrics['amplitude_ratio'], 'purple')
    amp_threshold = np.exp(np.mean(np.log(metrics['amplitude_ratio'] + 1e-10)) + 2.0 * np.std(np.log(metrics['amplitude_ratio'] + 1e-10)))
    plt.axhline(y=amp_threshold, color='r', linestyle='--', label='Threshold')
    plt.xlabel('Window index')
    plt.ylabel('Amplitude ratio (clean/noisy)')
    plt.title('Amplitude Analysis (Detects Attenuator)')
    
    plt.tight_layout()
    plt.savefig('advanced_anomaly_detection.png')
    plt.show()
    
    # Print detected anomalies
    print("\nDetected anomalies:")
    anomaly_points = []
    for idx in anomaly_indices:
        start_pos = test_indices[idx]
        end_pos = start_pos + seq_len
        anomaly_points.extend(range(start_pos, end_pos))
        print(f"Anomaly at positions {start_pos} to {end_pos}")
    
    # Check which known anomalies were detected
    known_anomalies = {
        "Point anomalies": [12, 111, 444],
        "Collective anomaly": list(range(100, 110)),
        "Frequency anomaly": list(range(700, 720)),
        "Phase shift anomaly": list(range(800, 820)),
        "Amplitude anomaly": list(range(900, 920))
    }
    
    print("\nKnown anomaly detection status:")
    for name, positions in known_anomalies.items():
        # Consider anomaly detected if any position in the anomaly range is detected
        detected = any(pos in anomaly_points for pos in positions)
        print(f"{name}: {'DETECTED' if detected else 'MISSED'}")

if __name__ == "__main__":
    main()