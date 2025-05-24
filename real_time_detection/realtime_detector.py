import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from collections import deque

from model_components import (
    _extract_raw_features_for_signal_segment,
    extract_autoencoder_input_vector_from_raw_features,
    AdvancedAutoencoder,
    AnomalyScorer
)

MODEL_DIR = 'trained_autoencoder'
MODEL_STATE_DICT_PATH = os.path.join(MODEL_DIR, "autoencoder_state_dict.pth")
TRAINING_METADATA_PATH = os.path.join(MODEL_DIR, "training_metadata.pkl")

def load_training_artifacts():
    if not os.path.exists(MODEL_STATE_DICT_PATH) or not os.path.exists(TRAINING_METADATA_PATH):
        raise FileNotFoundError("Trained model data not found. To run real-time detection, please train the model first.")
    
    with open(TRAINING_METADATA_PATH, 'rb') as f:
        metadata = pickle.load(f)
        
    scaler = metadata['scaler']
    input_size = metadata['input_size']
    seq_len = metadata['seq_len']
    overlap = metadata['overlap'] # overlap is not directly used in detection but parameters seq_len.
    thresholds = metadata['threshold']
    
    model = AdvancedAutoencoder(input_size)
    model.load_state_dict(torch.load(MODEL_STATE_DICT_PATH))
    model.eval()
    
    print(f"\nLoaded model and scaler from {MODEL_DIR}.\n")
    return model, scaler, seq_len, thresholds




class RealtimeAnomalyDetector:
    def __init__(self, model, scaler, seq_len, anomaly_scorer):
        self.model = model
        self.scaler = scaler
        self.seq_len = seq_len
        self.anomaly_scorer = anomaly_scorer
        self.noisy_buffer = deque(maxlen=seq_len)
        self.clean_buffer = deque(maxlen=seq_len)
        
        self.detected_anomalies_start_indices = []
        
        self.metrics_history = {
            'reconstruction_error': [],
            'frequency_deviation': [],
            'phase_deviation': [],
            'amplitude_deviation': []
        }
        self.processed_segment_start_indices = []
        
    def process_new_data_point(self, current_noisy_point, current_clean_point, current_point_global_index):
        self.noisy_buffer.append(current_noisy_point)
        self.clean_buffer.append(current_clean_point)
        
        if len(self.noisy_buffer) == self.seq_len:
            noisy_segment = np.array(list(self.noisy_buffer), dtype=np.float32)
            clean_segment = np.array(list(self.clean_buffer), dtype=np.float32)
        
            noisy_raw_features = _extract_raw_features_for_signal_segment(noisy_segment)
            clean_raw_features = _extract_raw_features_for_signal_segment(clean_segment)
            
            if not noisy_raw_features or not clean_raw_features:
                print("Insufficient features extracted from the segment. Skipping this segment.")
                return
            
            # Extract autoencoder input vectors from raw features
            noisy_ae_features = extract_autoencoder_input_vector_from_raw_features(noisy_raw_features)
            noisy_ae_features_scaled = self.scaler.transform(noisy_ae_features.reshape(1, -1))
            
            # Autoencoder reconstruction
            with torch.no_grad():
                reconstructed_ae_features_scaled = self.model(torch.tensor(noisy_ae_features_scaled, dtype=torch.float32)).numpy()
            
            # Evaluate the segment using the anomaly scorer
            results = self.anomaly_scorer.evaluate_segment(
                noisy_ae_features_scaled,
                reconstructed_ae_features_scaled,
                clean_raw_features,
                noisy_raw_features
            )
            
            self.metrics_history['reconstruction_error'].append(results['reconstruction_error']['score'])
            self.metrics_history['frequency_deviation'].append(results['frequency_deviation']['score'])
            self.metrics_history['phase_deviation'].append(results['phase_deviation']['score'])
            self.metrics_history['amplitude_deviation'].append(results['amplitude_deviation']['score'])
            
            segment_start_index = current_point_global_index - self.seq_len + 1
            self.processed_segment_start_indices.append(segment_start_index)

            if results['is_any_anomaly']:
                self.detected_anomalies_start_indices.append(segment_start_index)
                print(f"Anomaly detected in a segment from {segment_start_index} to {current_point_global_index}. Result: {results}")

def main():
    # Base signal for comparison
    x = np.linspace(0, 10 * np.pi, 1000)
    y_clean = np.abs(np.sin(x)).astype(np.float32)
    
    y_noisy = y_clean.copy()
    
    # Examples of anomalies
    y_noisy[12] -= 0.8
    y_noisy[444] += 0.7
    y_noisy[111] += 1.0
    y_noisy[100:110] += 0.5

    x_3x = x[700:720]
    y_noisy[700:720] = np.abs(np.sin(3 * x_3x)).astype(np.float32)  # 3x frequency
    x_plus_phase = x[800:820]
    y_noisy[800:820] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)  # phase shift
    x_attenuator = x[900:920]
    y_noisy[900:920] = np.abs(np.sin(x_attenuator) * 0.1).astype(np.float32)  # attenuator

    print("\n--- Start of real time detection ---")
    
    model, train_scaler, seq_len, thresholds = load_training_artifacts()
    
    anomaly_scorer = AnomalyScorer(
        recon_error_threshold_params=thresholds['reconstruction_error'],
        freq_deviation_threshold_params=thresholds['frequency_deviation'],
        phase_deviation_threshold_params=thresholds['phase_deviation'],
        amp_ratio_threshold_params=thresholds['amplitude_deviation']
    )
    
    realtime_detector = RealtimeAnomalyDetector(model, train_scaler, seq_len, anomaly_scorer)
    
    for i in range(len(y_noisy)):
        realtime_detector.process_new_data_point(y_noisy[i], y_clean[i], i)
        
    print("\n--- End of real time detection. Results visualization ---")
    
    plt.figure(figsize=(15, 15))
    
    plt.subplot(4, 1, 1)
    plt.plot(y_clean, 'g-', alpha=0.5, label='Czysty sygnał')
    plt.plot(y_noisy, 'b-', label='Zaszumiony sygnał')
    
    # Light up detected anomalies
    for start_idx in realtime_detector.detected_anomalies_start_indices:
        plt.axvspan(start_idx, start_idx + seq_len, color='red', alpha=0.3)
    
    # Light up true anomalies for comparison
    plt.axvspan(700, 720, color='yellow', alpha=0.2, label='Anomalia częstotliwości (3x)')
    plt.axvspan(800, 820, color='orange', alpha=0.2, label='Anomalia przesunięcia fazy')
    plt.axvspan(900, 920, color='purple', alpha=0.2, label='Anomalia tłumienia amplitudy')
    plt.legend()
    plt.title('Sygnał z wykrytymi anomaliami w czasie rzeczywistym')
    
    # Chart 2: Reconstruction error
    plt.subplot(4, 1, 2)
    plt.plot(realtime_detector.processed_segment_start_indices, realtime_detector.metrics_history['reconstruction_error'], 'b-')
    re_thresh = realtime_detector.anomaly_scorer.re_mean + realtime_detector.anomaly_scorer.re_sensitivity * realtime_detector.anomaly_scorer.re_std
    plt.axhline(y=re_thresh, color='r', linestyle='--', label='Próg')
    plt.xlabel('Indeks początku segmentu')
    plt.ylabel('Błąd rekonstrukcji')
    plt.title('Analiza błędu rekonstrukcji (czas rzeczywisty)')
    
    # Chart 3: Frequency deviation
    plt.subplot(4, 1, 3)
    plt.plot(realtime_detector.processed_segment_start_indices, realtime_detector.metrics_history['frequency_deviation'], 'g-')
    fd_thresh = realtime_detector.anomaly_scorer.fd_mean + realtime_detector.anomaly_scorer.fd_sensitivity * realtime_detector.anomaly_scorer.fd_std
    plt.axhline(y=fd_thresh, color='r', linestyle='--', label='Próg')
    plt.xlabel('Indeks początku segmentu')
    plt.ylabel('Odchylenie częstotliwości')
    plt.title('Analiza częstotliwości (czas rzeczywisty)')
    
    # Chart 4: Amplitude deviation
    plt.subplot(4, 1, 4)
    plt.plot(realtime_detector.processed_segment_start_indices, realtime_detector.metrics_history['amplitude_deviation'], 'purple')
    ar_thresh = realtime_detector.anomaly_scorer.ar_mean + realtime_detector.anomaly_scorer.ar_sensitivity * realtime_detector.anomaly_scorer.ar_std
    plt.axhline(y=ar_thresh, color='r', linestyle='--', label='Próg')
    plt.xlabel('Indeks początku segmentu')
    plt.ylabel('Odchylenie amplitudy')
    plt.title('Analiza amplitudy (czas rzeczywisty)')
    
    plt.tight_layout()
    plt.savefig('realtime_anomaly_detection.png')
    plt.show()

    # write up detected anomalies
    print("\nWykryte segmenty anomalne:")
    anomaly_points_set = set()
    for start_idx in realtime_detector.detected_anomalies_start_indices:
        end_idx = start_idx + seq_len - 1
        print(f"Segment od {start_idx} do {end_idx}")
        for p in range(start_idx, end_idx + 1):
            anomaly_points_set.add(p)

    # checking of what anomalies were detected
    known_anomalies = {
        "Anomalie punktowe (12, 111, 444)": [12, 111, 444],
        "Anomalia zbiorcza (100-109)": list(range(100, 110)),
        "Anomalia częstotliwości (700-719)": list(range(700, 720)),
        "Anomalia przesunięcia fazy (800-819)": list(range(800, 820)),
        "Anomalia tłumienia amplitudy (900-919)": list(range(900, 920))
    }
    
    print("\nDetected anomalies summary:")
    for name, positions in known_anomalies.items():
        detected_count = sum(1 for pos in positions if pos in anomaly_points_set)
        if len(positions) > 0:
            detection_percentage = detected_count / len(positions)
            print(f"{name}: {'DETECTED' if detection_percentage >= 0.1 else 'MISSED'} (Detected {detected_count}/{len(positions)} points)")
        else:
            print(f"{name}: No points to check for detection.")
            
if __name__ == "__main__":
    main()