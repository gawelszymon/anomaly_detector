import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# autoencoder jest to sieć neuronowa, która uczy się kompresować dane wejściowe do mniejszej reprezentacji 
# (kodowania) i następnie rekonstruuje je z tej reprezentacji (dekodowanie).
# autoencoder jest używany do wykrywania anomalii, ponieważ uczy się normalnych
# wzorców w danych i może zidentyfikować dane, które są znacznie różne od tych wzorców jako anomalie.

# autoencoder jest unsupervised learning, ponieważ nie wymaga etykietowanych danych do nauki.
# unsupervised learning działa poprzez znajodowanie wzorców na podstawie poprawnych danych.

x = np.linspace(0, 10 * np.pi, 1000) # stworzenie dziedziny funkcji sinus
y_clean = np.abs(np.sin(x)).astype(np.float32)  # nałożenie na dziedzinę zbioru wartosci dla abs(sinx)

y_noisy = y_clean.copy()    # nalożenie anomali na pusta funkcje sinx
y_noisy[12] -= 0.8
y_noisy[444] += 0.7
y_noisy[111] += 1.0
y_noisy[100:110] += 0.5


def seq(data, seq_len): # przygotowuje dane do szeregów czasowych, dla modeli sekwencyjnych takich jak autoencoder
                        # date - ilosc pojedynczej sekwencji czyli ile punktów w czasie ma zawierac probka
                        # kazda sekwencja to jeden przyklad treningowy
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return torch.tensor(X).unsqueeze(-1)    # zamieniemy nasza tablice na tensor o wymiarach (ilosc sekwencji x dl sekwencji)
                                            # oraz dodajemy trzeci wymiar jakim jest funkcja dla kazdego pkt w kazdej sekwnecji
                                            # poprzez dodanie trzeciego wymiaru kazda probka ma taki jakby swoj kanal
                                            
    # tensor moze byc przetwarzany na GPU w tzw batach, czyli takich partiach danych, co bardzo przyspiesza operacje macierzowe
    
    # tensory wykorzystywane sa w uczeniu maszynowym i w deep leariningu poprzez wykonywanie na nich operacji macierzowych
    

# przekonwertowanie naszych funkcji na tensory
X_train = seq(y_clean, 20)
X_test = seq(y_noisy, 20)


# czyli ogólnie nasz model może nauczyć się trudniejszych rzeczy
# Linera przekształca liczby w inne liczby za pomocą wag i biasów, a
# ReLu dodaje nieliniowość do modelu, co pozwala na nauczenie się
# bardziej złożonych rzeczy
class AnomalyAutoencoder(nn.Module):    #klasa dziedziczaca po nn.Module jest modelem sieci neuronowej
    def __init__(self):
        super().__init__()  # wywołanie konstruktora klasy bazowej nn.Module
        self.encoder = nn.Sequential(
            nn.Linear(20, 10),  # pozwala zmienic dane z innego wymiaru na inny, czyli zmieniejszenie wymiarów pozwala na wydorebeinie najwanznieszjych cech danego zbioru przez co model sie go uczy
            nn.ReLU(),  # funkcja aktywacji, ktora wprowadza nieliniowosc do modelu, pozwala na nauczenie się przez model bardziej złożonyuch, rzeczy, dla wartosci ujemnych zwraca 0, dla innych zwraca to samo
            nn.Linear(10, 4)    # zmniejszamy wymiar, aby wyodrenic z niego jeszcze wazniejsz rzeczy, i sprawic zeby model mogł nauczysc sie jeszcze bardizej ogolnego wzorca
        )
        
        # odbudowanie danych z małej, ściętej formy do pełnej postaci
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