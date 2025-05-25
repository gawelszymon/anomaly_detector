import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft, signal
from sklearn.preprocessing import StandardScaler

x = np.linspace(0, 10 * np.pi, 1000)
y_clean = np.abs(np.sin(x)).astype(np.float32)

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
    

def main():
    
    # prepare data for the autoencoder
    seq_len = 20    #sample quantity - the data length of the segment
    # X_test_features - the features of the noisy signal (vector of the features)
    # train_scalar = the object StandardScaler to normalize the data (remember the mean and std of the training data)
    # train_indices - the beginning index of each segment, to map on the orignal signal
    X_train_features, train_scaler, train_indices = extract_advanced_features(y_clean, seq_len, overlap=0.75)
    
    # Prepare tensors to use in pytorch
    X_train = torch.tensor(X_train_features, dtype=torch.float32)
    
    # train the model
    input_size = X_train.shape[1]
    print(input_size)
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training with gradient clipping and early stopping
    epochs = 10000
    # patience = 100
    # best_loss = float('inf')
    # patience_counter = 0
    
    # the regural training loop
    for epoch in range(epochs):
        # the model is in training mode it takes input and learn the features returning the output
        output = model(X_train)
        loss = criterion(output, X_train)   # it computes the loss between the orginal and the reconstructed signal
        
        # Backward pass with gradient clipping, it updates the weights of the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping, close the gradient to 1.0
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
        
    torch.save(model.state_dict(), 'advanced_autoencoder.pth')

if __name__ == "__main__":
    main()