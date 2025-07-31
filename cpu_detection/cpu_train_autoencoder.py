import joblib
import numpy as np
import torch
import torch.nn as nn
from cpu_db_utils import fetch_train_data
from cpu_detection_components import (AdvancedAutoencoder,
                                    extract_advanced_features)

MODEL_PATH = 'advanced_autoencoder_cpu.pth'
SCALER_PATH = 'train_scaler_cpu.pkl'

def main():
    y_clean_cpu, users_data, time_sin_data, time_cos_data, _ = fetch_train_data()

    if len(y_clean_cpu) == 0:
        print("No training data available. Exiting.")
        return

    print(f"Loaded {len(y_clean_cpu)} CPU records for training.")

    seq_len = 30
    overlap = 0.5

    X_train_features, train_scaler, train_indices = extract_advanced_features(
        data_cpu=y_clean_cpu,
        seq_len=seq_len,
        overlap=overlap,
        fit_scaler=True,
        users_data=users_data,
        time_sin_data=time_sin_data,
        time_cos_data=time_cos_data
    )

    if X_train_features.shape[0] == 0:
        print(y_clean_cpu)
        print(f"Failed to extract training features (not enough data: {len(y_clean_cpu)} < seq_len: {seq_len}). Exiting.")
        return

    print(f"Extracted {X_train_features.shape[0]} feature segments, each of dimension {X_train_features.shape[1]}.")

    joblib.dump(train_scaler, SCALER_PATH)
    print(f"Training scaler saved to {SCALER_PATH}")

    X_train = torch.tensor(X_train_features, dtype=torch.float32)

    input_size = X_train.shape[1]
    model = AdvancedAutoencoder(input_size)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    epochs = 5000
    best_loss = float('inf')
    patience = 100
    patience_counter = 0

    print(f"Starting model training for {epochs} epochs (input_size: {input_size})...")

    for epoch in range(epochs):
        model.train()
        output = model(X_train)
        loss = criterion(output, X_train)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

        if loss.item() < best_loss - 1e-6:
            best_loss = loss.item()
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1

        if loss.item() < 0.0001:
            print(f"Loss threshold reached ({loss.item():.6f}) at epoch {epoch}.")
            torch.save(model.state_dict(), MODEL_PATH)
            break

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}. Best loss: {best_loss:.6f}")
            epoch = epochs - 1
            break

    if epoch == epochs - 1:
        print(f"Training completed after {epochs} epochs. Final loss: {loss.item():.6f}")
        if loss.item() >= best_loss:
            print(f"Model with best loss {best_loss:.6f} saved as {MODEL_PATH}")
        else:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Model with final loss {loss.item():.6f} saved as {MODEL_PATH}")

if __name__ == "__main__":
    main()
