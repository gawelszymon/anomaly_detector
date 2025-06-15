import torch
import numpy as np
import joblib # for loading the scaler

from cpu_detection_components import extract_advanced_features, AdvancedAutoencoder, AnomalyDetectorEnsemble
from cpu_detection.cpu_db_utils import fetch_test_data, fetch_train_data # needed fetch_train_data for y_reference_clean
from cpu_train_autoencoder import MODEL_PATH, SCALER_PATH

def calculate_expected_cpu(users, time_sin, time_cos):
    # here we conduct an expected (healthy and perfect) CPU calculation based on users and time features
    return 0.001 * users + 10 * time_sin + 5 * time_cos

def real_time_anomaly_detection(y_detected_cpu, users_data, time_sin_data, time_cos_data,
                                db_anomaly_labels, y_reference_clean_cpu, seq_len=30, overlap=0.5):
    # y_detected_cpu: cpus signal that is analyzed for anomalies
    # users_data, time_sin_data, time_cos_data: datas that make up the cpu
    # db_anomaly_labels: labels for test and validdation set that sey if the analized signal batch is anomaly or not
    # y_reference_clean_cpu: data from the trainging set that is used as a reference for ensemble detectors
    
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

    # preparation test data for autoencoder
    # fit_scaler=False, we use previously trained scaler
    X_test_features, _, test_indices = extract_advanced_features(
        y_detected_cpu, seq_len, overlap=overlap, scaler=scaler, fit_scaler=False
    )

    if X_test_features.shape[0] == 0:
        print("Test features extraction returned no segments. Check the input data or parameters.")
        return

    print(f"Extract {X_test_features.shape[0]} segments of test features, each of dimension: {X_test_features.shape[1]}.")
    
    # tensor preparation
    X_test = torch.tensor(X_test_features, dtype=torch.float32)
    
    # model loading
    input_size = X_test.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"The trained model not found {MODEL_PATH}. Run the training process.")
        return
    except Exception as e:
        print(f"ERROR: Model not run properly: {e}")
        return
        
    model.eval() # set model to evaluation mode, so we tell that the model is not in training mode anymore, so we turn off dropout and batch normalization
    
    # here the model is used to reconstruct the features, it is made in special context torch.no_grad()
    with torch.no_grad(): # we can turn off gradients, because we are not training the model, so we weights are not updated
        reconstructed_features = model(X_test)
    
    # this kind of detector represents the technique that is responsible for merging the results of different detectors to get better detection results
    ensemble = AnomalyDetectorEnsemble(X_test, reconstructed_features, test_indices, seq_len)
    
    # y_reference_clean_cpu is trainind data that are for reference
    # y_detected_cpu is actualy detected CPU signal that is analyzed for anomalies
    anomaly_segment_indices, metrics = ensemble.run_all_detectors(y_reference_clean_cpu, y_detected_cpu)
    
    # print(f"\nDetector metrics: ")
    # print(f"  Mean Squared Error (as a reconstruction error) for the first five segments: {metrics['reconstruction_error'][:5]}")

    # anomaly indecs are converted into actual anomaly points
    detected_anomaly_points = set()
    if test_indices.size > 0 and len(anomaly_segment_indices) > 0:
        for segment_idx in anomaly_segment_indices:
            if segment_idx < len(test_indices):
                start_pos = test_indices[segment_idx]
                end_pos = start_pos + seq_len
                detected_anomaly_points.update(range(start_pos, end_pos))
    
    print(f"\nDetected {len(detected_anomaly_points)} data points as anomalies (from {len(y_detected_cpu)}).")
    if len(detected_anomaly_points) > 0 and len(detected_anomaly_points) < 20:
        print(f"  Points of anomalies indices: {sorted(list(detected_anomaly_points))[:20]}...")


    # Evaluation that bases on +/- 5% rule from expected CPU
    print("\n +/- 5% ewaluation:")
    anomalies_by_5_percent_rule_indices = []
    expected_cpu_values = calculate_expected_cpu(users_data, time_sin_data, time_cos_data)
    
    for i in range(len(y_detected_cpu)):
        expected_cpu = expected_cpu_values[i]
        actual_cpu = y_detected_cpu[i]
        
        if expected_cpu == 0: # for avoid division by zero
            if actual_cpu != 0:
                pass 
            continue

        deviation = np.abs(actual_cpu - expected_cpu) / np.abs(expected_cpu)
        if deviation > 0.05:
            anomalies_by_5_percent_rule_indices.append(i)
            
    print(f"It has found {len(anomalies_by_5_percent_rule_indices)} anomalies according to 5% rule.")
    if len(anomalies_by_5_percent_rule_indices) > 0 and len(anomalies_by_5_percent_rule_indices) < 20:
        print(f" 5% rule point's indeces: {anomalies_by_5_percent_rule_indices[:20]}...")

    # Porównanie wykryć autoencodera z regułą 5%
    true_positives_5_percent = len(detected_anomaly_points.intersection(set(anomalies_by_5_percent_rule_indices)))
    false_positives_5_percent = len(detected_anomaly_points.difference(set(anomalies_by_5_percent_rule_indices)))
    false_negatives_5_percent = len(set(anomalies_by_5_percent_rule_indices).difference(detected_anomaly_points))

    precision_5_percent = true_positives_5_percent / (true_positives_5_percent + false_positives_5_percent) if (true_positives_5_percent + false_positives_5_percent) > 0 else 0
    recall_5_percent = true_positives_5_percent / (true_positives_5_percent + false_negatives_5_percent) if (true_positives_5_percent + false_negatives_5_percent) > 0 else 0
    f1_score_5_percent = 2 * (precision_5_percent * recall_5_percent) / (precision_5_percent + recall_5_percent) if (precision_5_percent + recall_5_percent) > 0 else 0
    
    print(f"  Autoencoder vs. Reguła 5%:")
    print(f"    TP: {true_positives_5_percent}, FP: {false_positives_5_percent}, FN: {false_negatives_5_percent}")
    print(f"    Precyzja: {precision_5_percent:.2f}, Czułość (Recall): {recall_5_percent:.2f}, F1-Score: {f1_score_5_percent:.2f}")

    # Porównanie z etykietami z bazy danych (jeśli dostępne)
    if db_anomaly_labels.size > 0 :
        db_anomaly_indices = np.where(db_anomaly_labels == 1)[0]
        print(f"\nEwaluacja na podstawie etykiet z bazy danych ({len(db_anomaly_indices)} anomalii w DB):")
        
        true_positives_db = len(detected_anomaly_points.intersection(set(db_anomaly_indices)))
        false_positives_db = len(detected_anomaly_points.difference(set(db_anomaly_indices)))
        false_negatives_db = len(set(db_anomaly_indices).difference(detected_anomaly_points))

        precision_db = true_positives_db / (true_positives_db + false_positives_db) if (true_positives_db + false_positives_db) > 0 else 0
        recall_db = true_positives_db / (true_positives_db + false_negatives_db) if (true_positives_db + false_negatives_db) > 0 else 0
        f1_score_db = 2 * (precision_db * recall_db) / (precision_db + recall_db) if (precision_db + recall_db) > 0 else 0

        print(f"  Autoencoder vs. Etykiety DB:")
        print(f"    TP: {true_positives_db}, FP: {false_positives_db}, FN: {false_negatives_db}")
        print(f"    Precyzja: {precision_db:.2f}, Czułość (Recall): {recall_db:.2f}, F1-Score: {f1_score_db:.2f}")


def main():
    # Pobierz dane testowe i referencyjne dane treningowe (dla ensemble)
    test_cpu, users, time_sin, time_cos, db_labels = fetch_test_data()
    train_cpu_ref = fetch_train_data() # Używane jako y_reference_clean_signal

    if test_cpu.size == 0:
        print("Brak danych testowych do analizy. Zakończono.")
        return
    if train_cpu_ref.size == 0:
        print("Brak referencyjnych danych treningowych CPU. Niektóre detektory w ensemble mogą nie działać poprawnie.")
        # Można kontynuować, ale z ostrzeżeniem, lub zakończyć
        # return 

    print(f"Pobrano {len(test_cpu)} rekordów testowych CPU.")
    print(f"Pobrano {len(train_cpu_ref)} rekordów treningowych CPU jako referencję.")
    
    # Możesz chcieć podzielić dane testowe na mniejsze partie, jeśli są bardzo duże
    # Tutaj przetwarzamy całość naraz
    real_time_anomaly_detection(test_cpu, users, time_sin, time_cos, db_labels, train_cpu_ref)

if __name__ == "__main__":
    main()