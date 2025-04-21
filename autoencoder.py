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


def seq(data, seq_len):
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return torch.tensor(X).unsqueeze(-1)

X_train = seq(y_clean, 20)
X_test = seq(y_noisy, 20)

class AnomalyAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 20)
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.view(x.size(0), 20, 1)

model = AnomalyAutoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1000):
    output = model(X_train)
    loss = criterion(output, X_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch} Loss {loss.item():.4f}")
        
with torch.no_grad():
    reconstructed = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructed)**2, dim=[1, 2]).numpy()

anomalies = np.where(reconstruction_error > (np.mean(reconstruction_error) + np.std(reconstruction_error)))[0]
print(anomalies)

# plt.figure(figsize=(10, 4))
# plt.plot(reconstruction_error, label='Reconstruction Error')
# plt.axhline(y=np.mean(reconstruction_error) + np.std(reconstruction_error), color='r', linestyle='--', label='Anomaly Threshold')
# plt.title("Anomaly Detection using Autoencoder")
# plt.legend()
# plt.show()