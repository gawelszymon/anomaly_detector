import torch
import torch.nn as nn
import numpy as np
import joblib

from cpu_detection_components import extract_advanced_features, AdvancedAutoencoder
from cpu_db_utils import fetch_train_data

MODEL_PATH = 'advanced_autoencoder_cpu.pth'
SCALER_PATH = 'train_scaler_cpu.pkl'

def main():
    y_clean_cpu, users_data, time_sin_data, time_cos_data = fetch_train_data()
    
    if len(y_clean_cpu) == 0:
        print("Brak danych treningowych. Zakończono.")
        return

    print(f"Pobrano {len(y_clean_cpu)} rekordów CPU do treningu.")

    seq_len = 30
    overlap = 0.5

    # *** FIX: Updated function call to match the new signature ***
    X_train_features, train_scaler, train_indices = extract_advanced_features(
        data_cpu=y_clean_cpu,
        seq_len=seq_len,
        overlap=overlap,
        fit_scaler=True,
        users_data=users_data,
        time_sin_data=time_sin_data,
        time_cos_data=time_cos_data
    )
    
    # This check is now robust and will catch the case where not enough data was available.
    if X_train_features.shape[0] == 0:
        print(y_clean_cpu)
        print(f"Nie udało się wyekstrahować cech treningowych (za mało danych: {len(y_clean_cpu)} < seq_len: {seq_len}). Zakończono.")
        return

    print(f"Wyekstrahowano {X_train_features.shape[0]} segmentów cech, każdy o wymiarze {X_train_features.shape[1]}.")

    joblib.dump(train_scaler, SCALER_PATH)
    print(f"Scaler treningowy zapisany w {SCALER_PATH}")
    
    X_train = torch.tensor(X_train_features, dtype=torch.float32)
    
    input_size = X_train.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 5000
    best_loss = float('inf')
    patience = 100
    patience_counter = 0

    print(f"Rozpoczynanie treningu modelu dla {epochs} epok (input_size: {input_size})...")
    
    for epoch in range(epochs):
        model.train()
        output = model(X_train)
        loss = criterion(output, X_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoka {epoch}, Strata: {loss.item():.6f}")
        
        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
        
        if loss.item() < 0.0001:
            print(f"Osiągnięto próg straty ({loss.item():.6f}) w epoce {epoch}.")
            torch.save(model.state_dict(), MODEL_PATH)
            break
        
        if patience_counter >= patience:
            print(f"Early stopping w epoce {epoch}. Najlepsza strata: {best_loss:.6f}")
            break
            
    if epoch == epochs - 1:
        print(f"Zakończono trening po {epochs} epokach. Ostateczna strata: {loss.item():.6f}")
        if loss.item() >= best_loss:
            print(f"Model z najlepszą stratą {best_loss:.6f} został zapisany jako {MODEL_PATH}")
        else:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model z ostateczną stratą {loss.item():.6f} zapisany jako {MODEL_PATH}")

if __name__ == "__main__":
    main()
