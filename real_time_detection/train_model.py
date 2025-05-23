import torch
import torch.nn as nn
import numpy as np
import pickle
import os

from model_components import (
    extract_features_for_batch_processing,
    _extract_raw_features_for_signal_segment,
    extract_autoencoder_input_vector_from_raw_features,
    AdvancedAutoencoder,
    AnomalyScorer
)

MODEL_DIR = 'trained_autoencoder'
MODEL_STATE_DICT_PATH = os.path.join(MODEL_DIR, "autoencoder_state_dict.pth")
TRAINING_METADATA_PATH = os.path.join(MODEL_DIR, "training_metadata.pkl")

def save_training_artifacts(model, scaler, input_size, seq_len, overlap, threshold):
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    torch.save(model.state_dict(), MODEL_STATE_DICT_PATH)
    
    metadata = {
        'scaler': scaler,
        'input_size': input_size,
        'seq_len': seq_len,
        'overlap': overlap,
        'threshold': threshold
    }
    with open(TRAINING_METADATA_PATH, 'wb') as f:
        pickle.dump(metadata, f)
        
    print(f"\nThe model and training metadata have been saved to {MODEL_DIR}.\n")
        
def main():
    x = np.linspace(0, 10 * np.pi, 1000)
    y_clean = np.abs(np.sin(x)).astype(np.float32)
    
    seq_len = 20
    overlap = 0.75
    
    X_train_features_scaled, train_scaler, train_indices = extract_features_for_batch_processing(y_clean, seq_len, overlap)
    
    X_train_tensor = torch.tensor(X_train_features_scaled, dtype=torch.float32)
    input_size = X_train_tensor.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epoches = 10000
    patience = 100
    best_loss = float('inf')
    patience_counter = 0
    
    print("\nStarting the training process...\n")
    for epoch in range(epoches):
        model.train()
        output = model(X_train_tensor)
        loss = criterion(output, X_train_tensor)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
            
        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}, due to no improvement in loss for {patience} epochs.")
                break
            
        if loss.item() < 0.0001:
            print(f"Early stopping at epoch {epoch}, due to loss below 0.0001.")
            break
        
    print(f"\nTraining completed. Computing the thresholds for anomaly detection...\n")
    
    step_for_metrics = int(seq_len * (1 - overlap))
    
    reconstruction_errors_clean = []
    frequency_deviations_clean = []
    phase_deviations_clean = []
    amplitude_ratios_clean = []
    
    with torch.no_grad():
        model.eval()
        
        reconstructed_clean_scaled = model(X_train_tensor).numpy()
        reconstruction_errors_clean = np.mean((X_train_features_scaled - reconstructed_clean_scaled) ** 2, axis=1)
        
        temp_scorer = AnomalyScorer(
            recon_error_threshold_params=(0,1,0),
            freq_deviation_threshold_params=(0,1,0),
            phase_deviation_threshold_params=(0,1,0),
            amp_ratio_threshold_params=(0,1,0)
        )
        
        for i in range(0, len(y_clean) - seq_len + 1, step_for_metrics):
            clean_segment = y_clean[i:i + seq_len]
            
            clean_raw_features = _extract_raw_features_for_signal_segment(clean_segment)
            
            if clean_raw_features:
                current_dom_freq = clean_raw_features.get('dominant_freq', 0)
                
                # Collecting the actual values for the clean signal, that is the baseline for comparison with the noisy signal
                if 'dominant_freq' in clean_raw_features:
                    frequency_deviations_clean.append(clean_raw_features['dominant_freq'])
                if 'phase_mean' in clean_raw_features:
                    phase_deviations_clean.append(clean_raw_features['phase_mean'])
                if 'std' in clean_raw_features:
                    amplitude_ratios_clean.append(np.log(clean_raw_features['std'] + 1e-10))
                    
    re_mean = np.mean(reconstruction_errors_clean)
    re_std = np.std(reconstruction_errors_clean)
    
    fd_diffs_from_mean_clean = np.abs(np.array(frequency_deviations_clean) - np.mean(frequency_deviations_clean))
    pd_diffs_from_mean_clean = np.abs(np.array(phase_deviations_clean) - np.mean(phase_deviations_clean))
    ar_diffs_from_mean_clean = np.abs(np.array(amplitude_ratios_clean) - np.mean(amplitude_ratios_clean))
    
    fd_mean_clean_diff = np.mean(fd_diffs_from_mean_clean)
    fd_std_clean_diff = np.std(fd_diffs_from_mean_clean)

    pd_mean_clean_diff = np.mean(pd_diffs_from_mean_clean)
    pd_std_clean_diff = np.std(pd_diffs_from_mean_clean)
    
    ar_mean_clean_diff = np.mean(ar_diffs_from_mean_clean)
    ar_std_clean_diff = np.std(ar_diffs_from_mean_clean)
    
    
    # Defintion of the sensitivity for the thresholds, how many standard deviations from the mean is considered an anomaly
    re_sensitivity = 2.5
    fd_sensitivity = 3.0
    pd_sensitivity = 2.0
    ar_sensitivity = 2.0

    thresholds = {
        'reconstruction_error': (re_mean, re_std, re_sensitivity),
        'frequency_deviation': (fd_mean_clean_diff, fd_std_clean_diff, fd_sensitivity),
        'phase_deviation': (pd_mean_clean_diff, pd_std_clean_diff, pd_sensitivity),
        'amplitude_deviation': (ar_mean_clean_diff, ar_std_clean_diff, ar_sensitivity)
    }
    
    save_training_artifacts(model, train_scaler, input_size, seq_len, overlap, thresholds)
    
if __name__ == "__main__":
    main()