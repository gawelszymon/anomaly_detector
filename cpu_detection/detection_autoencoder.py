import torch
import numpy as np


from detection_components import extract_advanced_features, AdvancedAutoencoder, AnomalyDetectorEnsemble

x = np.linspace(0, 10 * np.pi, 1000)
y_clean = np.abs(np.sin(x)).astype(np.float32)
y_noisy = y_clean.copy()

# common anomalies
y_noisy[70] -= 0.8
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


y_noisy1 = y_clean.copy()

# common anomalies
y_noisy1[120] -= 0.8
y_noisy1[333] += 0.7
y_noisy1[222] += 1.0
y_noisy1[100:110] += 0.5

# silient anomalies
x_3x = x[500:520]
y_noisy1[500:520] = np.abs(np.sin(3 * x_3x)).astype(np.float32)  # 3x frequency
x_plus_phase = x[850:870]
y_noisy1[850:870] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)  # phase shift
x_attenuator = x[970:990]
y_noisy1[970:990] = np.abs(np.sin(x_attenuator) * 0.1).astype(np.float32)  # attenuator


y_noisy2 = y_clean.copy()
y_noisy2[111] += 1.7
x_plus_phase = x[850:870]
y_noisy2[850:870] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)

y_noisy3 = y_clean.copy()
y_noisy3[111] += 1.7
x_plus_phase = x[300:380]
y_noisy3[300:380] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)

detector = []
detector.extend([y_noisy, y_noisy1, y_noisy2, y_noisy3])

def real_time_anomaly_detection(y_detected):
    # prepare data for the autoencoder
    seq_len = 20    #sample quantity - the data length of the segment
    # X_test_features - the features of the noisy signal (vector of the features)
    # test_indices - the beginning index of each segment, to map on the orignal signal
    
    X_test_features, _, test_indices = extract_advanced_features(y_detected, seq_len, overlap=0.75)
    
    # Prepare tensors to use in pytorch
    X_test = torch.tensor(X_test_features, dtype=torch.float32)
    
    input_size = X_test.shape[1] #73
    model = AdvancedAutoencoder(input_size)
    
    model.load_state_dict(torch.load('advanced_autoencoder.pth'))
    model.eval()
    
    # Evaluate model and detect anomalies
    with torch.no_grad():
        reconstructed = model(X_test)
    
    # Use ensemble detector for better detection of different anomaly types
    ensemble = AnomalyDetectorEnsemble(X_test, reconstructed, test_indices, seq_len)
    anomaly_indices, metrics = ensemble.run_all_detectors(y_clean, y_detected)
    
    # Print detected anomalies
    print("\nDetected anomalies:")
    anomaly_points = []
    for idx in anomaly_indices:
        start_pos = test_indices[idx]
        end_pos = start_pos + seq_len
        anomaly_points.extend(range(start_pos, end_pos))
        print(f"Anomaly at positions {start_pos} to {end_pos}")
        
    # # Check which known anomalies were detected
    # known_anomalies = {
    #     "Point anomalies": [12, 111, 444],
    #     "Collective anomaly": list(range(100, 110)),
    #     "Frequency anomaly": list(range(700, 720)),
    #     "Phase shift anomaly": list(range(800, 820)),
    #     "Amplitude anomaly": list(range(900, 920))
    # }
    
    # print("\nKnown anomaly detection status:")
    # for name, positions in known_anomalies.items():
    #     # Consider anomaly detected if any position in the anomaly range is detected
    #     detected = any(pos in anomaly_points for pos in positions)
    #     print(f"{name}: {'DETECTED' if detected else 'MISSED'}")

def main():
    for noisy in detector:
        real_time_anomaly_detection(noisy)
        print(f"\n---------------------------\n")

if __name__ == "__main__":
    main()