import torch
import torch.nn as nn
import numpy as np
import joblib # Do zapisywania scalera

from cpu_detection_components import extract_advanced_features, AdvancedAutoencoder
from cpu_detection.cpu_db_utils import fetch_train_data

MODEL_PATH = 'advanced_autoencoder_cpu.pth'
SCALER_PATH = 'train_scaler_cpu.pkl'

def main():
    # Pobierz dane treningowe CPU z bazy danych
    y_clean_cpu = fetch_train_data()
    
    if y_clean_cpu.size == 0:
        print("Brak danych treningowych. Zakończono.")
        return

    print(f"Pobrano {len(y_clean_cpu)} rekordów CPU do treningu.")

    # Przygotuj dane dla autoencodera
    seq_len = 30  # Długość segmentu, można dostosować
    overlap = 0.5 # Stopień nakładania się segmentów

    # fit_scaler=True, aby stworzyć i dopasować nowy scaler
    X_train_features, train_scaler, train_indices = extract_advanced_features(
        y_clean_cpu, seq_len, overlap=overlap, fit_scaler=True
    )
    
    if X_train_features.shape[0] == 0:
        print("Nie udało się wyekstrahować cech treningowych (możliwe, że za mało danych). Zakończono.")
        return

    print(f"Wyekstrahowano {X_train_features.shape[0]} segmentów cech, każdy o wymiarze {X_train_features.shape[1]}.")

    # Zapisz scaler
    joblib.dump(train_scaler, SCALER_PATH)
    print(f"Scaler treningowy zapisany w {SCALER_PATH}")
    
    # Przygotuj tensory dla PyTorch
    X_train = torch.tensor(X_train_features, dtype=torch.float32)
    
    # Trenuj model
    input_size = X_train.shape[1]
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    epochs = 5000 # Można dostosować; dla CPU może być potrzebne więcej lub mniej
    best_loss = float('inf')
    patience = 100 # Dla early stopping
    patience_counter = 0

    print(f"Rozpoczynanie treningu modelu dla {epochs} epok (input_size: {input_size})...")
    
    for epoch in range(epochs):
        model.train() # Ustaw model w tryb treningowy
        output = model(X_train)
        loss = criterion(output, X_train)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoka {epoch}, Strata: {loss.item():.6f}")
        
        # Early stopping
        if loss.item() < best_loss - 1e-6: # Minimalna poprawa
            best_loss = loss.item()
            patience_counter = 0
            # Zapisz najlepszy model
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
        
        if loss.item() < 0.0001: # Próg straty
            print(f"Osiągnięto próg straty ({loss.item():.6f}) w epoce {epoch}.")
            torch.save(model.state_dict(), MODEL_PATH) # Zapisz ostateczny model
            break
        
        if patience_counter >= patience:
            print(f"Early stopping w epoce {epoch}. Najlepsza strata: {best_loss:.6f}")
            break # Model już został zapisany przy najlepszej stracie
            
    if epoch == epochs -1: # Jeśli pętla zakończyła się normalnie
        print(f"Zakończono trening po {epochs} epokach. Ostateczna strata: {loss.item():.6f}")
        # Upewnij się, że model jest zapisany, jeśli nie było early stopping i nie osiągnięto progu straty
        if loss.item() >= best_loss: # jeśli ostatnia strata nie była lepsza, najlepszy jest już zapisany
            print(f"Model z najlepszą stratą {best_loss:.6f} został zapisany jako {MODEL_PATH}")
        else: # ostatnia strata była najlepsza (lub jedyna zapisywana)
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model z ostateczną stratą {loss.item():.6f} zapisany jako {MODEL_PATH}")


if __name__ == "__main__":
    main()