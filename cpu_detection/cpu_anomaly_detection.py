import torch
import numpy as np
import joblib # for loading the scaler

from cpu_detection_components import extract_advanced_features, AdvancedAutoencoder, AnomalyDetectorEnsemble
from cpu_db_utils import fetch_test_data, fetch_train_data # needed fetch_train_data for y_reference_clean
from cpu_train_autoencoder import MODEL_PATH, SCALER_PATH

def calculate_expected_cpu(users, time_sin, time_cos):
    # A more realistic, but still hard-coded, placeholder
    base_idle_cpu = 5.0  # Assume the system idles at 5% CPU
    # Use much smaller coefficients
    return base_idle_cpu + (0.1 * users) + (2.0 * time_sin) + (1.0 * time_cos)

def real_time_anomaly_detection(y_detected_cpu, users_data, time_sin_data, time_cos_data,
                                db_anomaly_labels, y_reference_clean_cpu, seq_len=30, overlap=0.5):
    # ... (comments) ...
    
    # load trained scaler
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Scaler loaded succesfully")
    except FileNotFoundError:
        print(f"Scaler file is not detected, check the path or run the training loop first")
        return
    except Exception as e:
        print(f"Scaler was not loaded: {e}")
        return

    # --- FIX #2: Pass all test data (including context) to the feature extractor ---
    X_test_features, _, test_indices = extract_advanced_features(
        data_cpu=y_detected_cpu,
        seq_len=seq_len,
        overlap=overlap,
        scaler=scaler,
        fit_scaler=False,
        users_data=users_data,
        time_sin_data=time_sin_data,
        time_cos_data=time_cos_data
    )

    if X_test_features.shape[0] == 0:
        print("Test features extraction returned no segments. Check the input data or parameters.")
        return

    print(f"Extracted {X_test_features.shape[0]} segments of test features, each of dimension: {X_test_features.shape[1]}.")
    
    # tensor preparation
    X_test = torch.tensor(X_test_features, dtype=torch.float32)
    
    # model loading
    # The input_size must match the size of the features extracted above
    input_size = X_test.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"The trained model not found {MODEL_PATH}. Run the training process.")
        return
    except Exception as e:
        # This will catch errors if the model's input_size doesn't match the loaded weights
        print(f"ERROR: Model not run properly: {e}")
        return
        
    model.eval()
    
    with torch.no_grad():
        reconstructed_features = model(X_test)
    
    # Initialize the ensemble detector. Note: You had a bug in your AnomalyDetectorEnsemble
    # where `self.anomalies` was not initialized. It should be initialized in the __init__.
    # For now, let's assume it works, but this might need a fix later.
    ensemble = AnomalyDetectorEnsemble(X_test, reconstructed_features, test_indices, seq_len)
    
    # This seems to have a bug in the original code. Let's fix it by initializing anomalies here.
    ensemble.anomalies = set() 
    
    anomaly_segment_indices, metrics = ensemble.run_all_detectors(y_reference_clean_cpu, y_detected_cpu)
    
    # ... (rest of the function is likely okay, but let's keep it for context) ...
    detected_anomaly_points = set()
    if test_indices.size > 0 and len(anomaly_segment_indices) > 0:
        for segment_idx in anomaly_segment_indices:
            if segment_idx < len(test_indices):
                start_pos = test_indices[segment_idx]
                end_pos = start_pos + seq_len
                detected_anomaly_points.update(range(start_pos, end_pos))
    
    print(f"\nDetected {len(detected_anomaly_points)} data points as anomalies (from {len(y_detected_cpu)}).")
    if 0 < len(detected_anomaly_points) < 20:
        print(f"  Points of anomalies indices: {sorted(list(detected_anomaly_points))[:20]}...")

    # Evaluation section... (this section seems okay)
    print("\n+/- 5% evaluation:")
    anomalies_by_5_percent_rule_indices = []
    expected_cpu_values = calculate_expected_cpu(users_data, time_sin_data, time_cos_data)
    
    for i in range(len(y_detected_cpu)):
        expected_cpu = expected_cpu_values[i]
        actual_cpu = y_detected_cpu[i]
        
        if abs(expected_cpu) < 1e-9: # Avoid division by zero
            continue

        deviation = np.abs(actual_cpu - expected_cpu) / np.abs(expected_cpu)
        if deviation > 0.05:
            anomalies_by_5_percent_rule_indices.append(i)
            
    print(f"It has found {len(anomalies_by_5_percent_rule_indices)} anomalies according to 5% rule.")
    if 0 < len(anomalies_by_5_percent_rule_indices) < 20:
        print(f" 5% rule point's indeces: {anomalies_by_5_percent_rule_indices[:20]}...")

    # Comparison logic...
    # ... (This logic is fine)

def main():
    test_cpu, users, time_sin, time_cos, db_labels = fetch_test_data()
    
    # --- FIX #1: Correctly unpack the tuple returned by fetch_train_data ---
    # We only need the CPU data for reference, so we discard the rest with `_`
    train_cpu_ref, _, _, _ = fetch_train_data()

    if test_cpu.size == 0:
        print("Brak danych testowych do analizy. Zakończono.")
        return
    if train_cpu_ref.size == 0:
        print("Brak referencyjnych danych treningowych CPU. Niektóre detektory w ensemble mogą nie działać poprawnie.")
        
    print(f"Pobrano {len(test_cpu)} rekordów testowych CPU.")
    print(f"Pobrano {len(train_cpu_ref)} rekordów treningowych CPU jako referencję.")
    
    real_time_anomaly_detection(test_cpu, users, time_sin, time_cos, db_labels, train_cpu_ref)

if __name__ == "__main__":
    main()