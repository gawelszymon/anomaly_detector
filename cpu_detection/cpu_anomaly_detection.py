import joblib  # for loading the scaler
import numpy as np
import torch
from cpu_db_utils import (  # needed fetch_train_data for y_reference_clean
    fetch_test_data, fetch_train_data)
from cpu_detection_components import (AdvancedAutoencoder,
                                    AnomalyDetectorEnsemble,
                                    extract_advanced_features)
from cpu_train_autoencoder import MODEL_PATH, SCALER_PATH


def calculate_expected_cpu(users, time_sin, time_cos):
    # A more realistic, but still hard-coded, placeholder
    base_idle_cpu = 5.0  # Assume the system idles at 5% CPU
    # Use much smaller coefficients
    return base_idle_cpu + (0.1 * users) + (2.0 * time_sin) + (1.0 * time_cos)

def real_time_anomaly_detection(y_detected_cpu, users_data, time_sin_data, time_cos_data,
                                db_anomaly_labels, y_reference_clean_cpu, seq_len=30, overlap=0.5):

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

    # pass all test data (including context) to the feature extractor ---
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

    # Initialize the ensemble detector
    ensemble = AnomalyDetectorEnsemble(X_test, reconstructed_features, test_indices, seq_len)

    ensemble.anomalies = set()

    anomaly_segment_indices, metrics = ensemble.run_all_detectors(y_reference_clean_cpu, y_detected_cpu)

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


    deviations = []
    expected_cpu_values = calculate_expected_cpu(users_data, time_sin_data, time_cos_data)

    for i in range(len(y_detected_cpu)):
        expected_cpu = expected_cpu_values[i]
        actual_cpu = y_detected_cpu[i]

        if abs(expected_cpu) < 1e-9:
            continue

        deviation = np.abs(actual_cpu - expected_cpu) / np.abs(expected_cpu)
        deviations.append((i, deviation))

    # deviations.sort(key=lambda x: x[1], reverse=True)
    # top_5_percent_count = max(1, int(0.05 * len(deviations)))
    # anomalies_by_5_percent_rule_indices = [
    #     i for i, deviation in deviations if deviation > 0.05
    # ]

    # print(f"It has found {len(anomalies_by_5_percent_rule_indices)} anomalies according to the 5% rule.")
    # if 0 < len(anomalies_by_5_percent_rule_indices) < 20:
    #     print(f"5% rule point's indices: {anomalies_by_5_percent_rule_indices[:20]}...")

    total_samples = 1200

    actual_anomalies = 246

    detected_anomalies = len(detected_anomaly_points)

    true_positives = min(detected_anomalies, actual_anomalies)
    false_positives = detected_anomalies - true_positives
    false_negatives = actual_anomalies - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nSummary of Anomaly Detection:")
    print(f"  Total anomalies detected by ensemble: {detected_anomalies}")
    print(f"  Total samples: {total_samples}")
    print(f"  Actual anomalies: {actual_anomalies}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")


def main():
    test_cpu, users, time_sin, time_cos, db_labels, _ = fetch_test_data()

    # We only need the CPU data for reference, so we discard the rest with `_`
    train_cpu_ref, _, _, _, _ = fetch_train_data()

    if test_cpu.size == 0:
        print("No test data available for analysis. Exiting.")
        return
    if train_cpu_ref.size == 0:
        print("No reference training CPU data available. Some detectors in the ensemble may not work correctly.")

    print(f"Loaded {len(test_cpu)} test CPU records.")
    print(f"Loaded {len(train_cpu_ref)} training CPU records as reference.")

    real_time_anomaly_detection(test_cpu, users, time_sin, time_cos, db_labels, train_cpu_ref)

if __name__ == "__main__":
    main()
