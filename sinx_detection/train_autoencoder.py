import torch
import torch.nn as nn
import numpy as np

from detection_components import extract_advanced_features, AdvancedAutoencoder

x = np.linspace(0, 10 * np.pi, 1000)
y_clean = np.abs(np.sin(x)).astype(np.float32)


def main():
    
    # prepare data for the autoencoder
    seq_len = 20    #sample quantity - the data length of the segment
    # X_test_features - the features of the noisy signal (vector of the features)
    # train_scalar = the object StandardScaler to normalize the data (remember the mean and std of the training data)
    # train_indices - the beginning index of each segment, to map on the orignal signal
    X_train_features, train_scaler, train_indices = extract_advanced_features(y_clean, seq_len, overlap=0.75)
    
    # Prepare tensors to use in pytorch
    X_train = torch.tensor(X_train_features, dtype=torch.float32)
    
    # train the model
    input_size = X_train.shape[1]
    print(input_size)
    model = AdvancedAutoencoder(input_size)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training with gradient clipping and early stopping
    epochs = 10000
    # patience = 100
    # best_loss = float('inf')
    # patience_counter = 0
    
    # the regural training loop
    for epoch in range(epochs):
        # the model is in training mode it takes input and learn the features returning the output
        output = model(X_train)
        loss = criterion(output, X_train)   # it computes the loss between the orginal and the reconstructed signal
        
        # Backward pass with gradient clipping, it updates the weights of the model
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # gradient clipping, close the gradient to 1.0
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        

        if loss.item() < 0.0001:
            print(f"Loss threshold reached at epoch {epoch}")
            break
        
    torch.save(model.state_dict(), 'advanced_autoencoder.pth')

if __name__ == "__main__":
    main()