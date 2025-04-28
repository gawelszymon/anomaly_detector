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
        
    def forward(self, x):   # funkcja forward odpowiada za przepłwy, przepuszczenie danych przez siec
        x = x.view(x.size(0), -1) # usuwamy wydmiar, aby linear mogło prztworzyć dane
        encoded = self.encoder(x) # komperesujemy dane do 4 wartośći
        decoded = self.decoder(encoded) # odtwarzamy pierwotne dane z skompresowanej postaci
        return decoded.view(x.size(0), 20, 1) # dodajemy wymiar, zeby wyjscie wygladało tak jak wejście

model = AnomalyAutoencoder()
criterion = nn.MSELoss() # funkcja straty, średni błąd kwadratowy, mierzy jak bardzo wyjście modelu różni się od prawdziwych danych (oceniamy jak dobrze model działa)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # optymalizator Adam, który aktualizuje wagi modelu na podstawie błędu
# model.parameters() - bierze wszystkie parametry modelu, które będą aktualizowane przez optymalizator
# lr - learning rate, mówi jak dużym krokiem będziemy aktualizować wagi modelu, czyli definiuje jak bardzo model ma zmieniac swoje parametry po kazydym korku nauki, 
# czyli w naszym przypadku każdej interacji.


# aktualizowana wparametry to wagi i biasy
# wagi to liczby mówiące jak mocno dane wejście wpływa na inny wynik
# biasy to przesunięcia dodane do wyniku
# parametr = wagi + biasy
# wagi i biasy rozróżnia się dla każdego wejścia, modyfikuje je się w celu minimalizacji błędu reprezentowanego przez funkcję straty

for epoch in range(5000):   # jedna epoka oznacza jedno przejscie przez caly zestaw treningowy
    output = model(X_train) # przepuszczenie danych poprawnych przez model, w wyniku dostaje output czyli rekonstrukcje tych danych
    loss = criterion(output, X_train) # obliczam błąd, miedyz tym co moedel przewidzial a tym co powinien przewidziec, za pomoca średniej kwadratowej
    optimizer.zero_grad() # zeruje gradienty, aby nie dodawaly sie do siebie
    loss.backward() # dostosowanie wag i biasów (gradientów) na podstawie błędu, pytorch robie to automatycznie
    optimizer.step() # aktualizujemy wagi i biasy w naszym modelu, tak aby loss (błąd) był mniejszy
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")
        
with torch.no_grad():  # wyłączamy gradienty, bo nie trenujemy modelu zbiorem testowym
    reconstructed = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructed)**2, dim=[1, 2]).numpy() # wektor zawierajacy blad rekonsttrukcji dla kazdej próbki danych testowych
    # oblciczamy średni błąd kwadratowy pomiedyz danymi orginalnymi a tymi odtworzonymi przez model

anomalies = np.where(reconstruction_error > (np.mean(reconstruction_error) + 2*np.std(reconstruction_error)))[0]
# np.mean(reconstruction_error), średnia wartość błędu rekonstrukcji dla wszystkich próbek
# np.std(reconstruction_error), odchylenie standardowe błędu rekonstrukcji dla wszystkich próbek

print(anomalies)

for i in anomalies:
    print(f"Anomalous sample {i}: y_noisy[{i}:{i+20}]")
    
for a in anomalies:
    idx_range = range(a, a + 20)
    print(f"Anomaly detected in points: {list(idx_range)}")

# błąd rekonstrukcji jest niski dla danych które są podobne do danych treningowych, a wysoki dla anomalii
# czyli za pomocą średniej i odchylenia standardowego ustalam powyżej którego dane uznawane są za anomalie

# średnia wartość błędu rekonstrukcji mówi jak dobrze model radzi sobie z odtwarzeniem danych
# odchylenie standardowe błędu rekonstrukcji mówi jak bardzo błedy zróżnicowane są w całym zbiorze danych
# czyli jesli błąd rekonstrukcji konkretnej próbki jest większy od średniej błędu rekonstrukcji + 2*odchylenie standardowe błędu rekonstrukcji to tą próbkę traktuję jako anomalie
# dlatego nasz model jako traktuję argumenty, które należą do próbek z anomalią.


# czyli jeśli mam dane zgodne z normalnym wzorcem, autoencoder poradzi sobie z ich rekonstrukcją i błąd bedzie mały/znikomy,
# jeśli wprowadzone dane są nietypowe, model nie bedzie w stanie ich dobrze odtaworzyc, bład bedize duzy
# następnie średnią wartość błedu rekonsttukcji dla calego zbioru danych  porównuje z błędem rekonstrukcji dla konkretnej próbki,
# jeśli błąd rekosntrukcji dla konkrentej próbki jest wiekszy od sredniej wartosrci błedu rekonstrukcji + 2*odchylenie standardowe błedu rekonstrukcji
# to traktuje ja jako anomalię