import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 10 * np.pi, 1000)
y_clean = np.abs(np.sin(x)).astype(np.float32)

y_noisy = y_clean.copy()

y_noisy[12] -= 0.8
y_noisy[444] += 0.7
y_noisy[111] += 1.0
y_noisy[100:110] += 0.5

x_3x = x[700:720]
y_noisy[700:720] = np.abs(np.sin(3 * x_3x)).astype(np.float32)
x_plus_phase = x[800:820]
y_noisy[800:820] = np.abs(np.sin(x_plus_phase + 1)).astype(np.float32)
x_attenuator = x[900:920]
y_noisy[900:920] = np.abs(np.sin(x_attenuator) * 0.1).astype(np.float32)

def compute_derivative(data):
    derivative = np.diff(data, prepend=data[0])
    return derivative.astype(np.float32)

y_clean_derivative = compute_derivative(y_clean)
y_noisy_derivative = compute_derivative(y_noisy)

def seq(data, derivative, seq_len):
    X = []
    for i in range(len(data) - seq_len):
        segment = np.stack([data[i:i + seq_len], derivative[i:i + seq_len]], axis=-1)
        X.append(segment)
    return torch.tensor(X)

X_train = seq(y_clean, y_clean_derivative, 10)
X_test = seq(y_noisy, y_noisy_derivative, 10)

class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(10 * 2, 5),
            nn.ReLU(),
            nn.Linear(5, 3)
        )
        self.decoder = nn.Sequential(
            nn.Linear(3, 5),
            nn.ReLU(),
            nn.Linear(5, 10 * 2)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 10, 2)

model = AnomalyAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10000):
    output = model(X_train)
    loss_value = criterion(output[:, :, 0], X_train[:, :, 0])
    loss_derivative = criterion(output[:, :, 1], X_train[:, :, 1])
    loss = loss_value + loss_derivative*0.5
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")
    if loss < 0.0001:
        break
        
with torch.no_grad():
    reconstructed = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructed) ** 2, dim=(1, 2)).numpy()

# anomalies = np.where(reconstruction_error > (np.mean(reconstruction_error) + 2 * np.std(reconstruction_error)))[0]

threshold = np.percentile(reconstruction_error, 95)
anomalies = np.where(reconstruction_error > threshold)[0]

print(anomalies)

for i in anomalies:
    print(f"Anomalous sample {i}: y_noisy[{i}:{i+10}]")
    
for a in anomalies:
    idx_range = range(a, a + 10)
    print(f"Anomaly detected in points: {list(idx_range)}")