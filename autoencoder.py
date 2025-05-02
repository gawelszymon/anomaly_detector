import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# autoencoder is a neural network that learns to compress input data into a smaller representation as a
# encoding and then reconstructs it from that representation as a decoding.
# autoencoder is used for anomaly detection because it learns normal patterns in the data and can identify
# data that is significantly different from those patterns as anomalies.

# autoencoder is unsupervised learning, because it is not requairng labeled data to learn.
# unsupervised learning works by finding patterns in the data without any labels based on correct data.

x = np.linspace(0, 10 * np.pi, 1000) # create a range of values for sin function
y_clean = np.abs(np.sin(x)).astype(np.float32)  # put the values of abs(sinx) function into our range

y_noisy = y_clean.copy()    # put some anomalies onto out function
y_noisy[12] -= 0.8
y_noisy[444] += 0.7
y_noisy[111] += 1.0
y_noisy[100:110] += 0.5


def seq(data, seq_len): # time stamp preparing for sequence models such as autoencoder
                        # date - time points quantity in the one sequence
                        # each sequence is one training sample
    X = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
    return torch.tensor(X).unsqueeze(-1)    # convert our array into a tensor with dimensions (number of sequences x sequence length)
                                            # and add a third dimension, which represents the function value for each point in each sequence
                                            # by adding the third dimension, each sample effectively has its own channel
                                            
    # tensors can be processed on a GPU in so-called batches, which significantly speeds up matrix operations
    
    # tensors are used in machine learning and deep learning by performing matrix operations on them
    

# our function convertion into tensor
X_train = seq(y_clean, 20)
X_test = seq(y_noisy, 20)


# In general, our model can learn more complex patterns.
# Linear transforms numbers into other numbers using weights and biases,
# while ReLU introduces non-linearity to the model, allowing it to learn
# more intricate patterns.
class AnomalyAutoencoder(nn.Module):    # the class inherits from nn.Module is a model of the neural network
    def __init__(self):
        super().__init__()  # calling the constructor of the parent class nn.Module 
        self.encoder = nn.Sequential(
            nn.Linear(20, 10), # allows transforming data from one dimension to another, reducing dimensions enables extracting the most important features of a given dataset, which helps the model learn it
            nn.ReLU(), # Activation function that introduces non-linearity to the model, allowing it to learn more complex patterns. For negative values, it returns 0; for others, it returns the same value.
            nn.Linear(10, 4)    # shrinking the dimension to 4 values, which allows us to extract even more important features of the dataset
        )
        
        # data reconstruction from a small, compressed form to its full form
        self.decoder = nn.Sequential(
            nn.Linear(4, 10),
            nn.ReLU(),
            nn.Linear(10, 20)
        )
        
    def forward(self, x):   # forward function is responsible for passing data through the network
        x = x.view(x.size(0), -1) # deleting the third dimension, because we don't need it in the encoder (linear layer can only process 2D data)
        encoded = self.encoder(x) # encoding the data, compressing it into a smaller representation (4 values)
        decoded = self.decoder(encoded) # decoding the data, reconstructing it back to its original form (20 values)
        return decoded.view(x.size(0), 20, 1) # adding back the third dimension, so the output has the same shape as the input

model = AnomalyAutoencoder()
criterion = nn.MSELoss() # we mark the difference between model output and real data by loss function and mean square error, we now how model is doing
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adam optimizer, which updates the model's weights based on the error
# model.parameters() - takes all the model parameters that will be updated by the optimizer
# lr - learning rate, defines how large a step we take to update the model's weights, i.e., how much the model changes its parameters after each learning step (epoch),
# in our case, each iteration.

# updated parameters are weights and biases
# weights are numbers that indicate how strongly an input affects a specific output
# biases are offsets added to the result
# parameter = weights + biases
# weights and biases are distinguished for each input and are modified to minimize the error represented by the loss function

for epoch in range(5000):   # one epoch means one pass through the entire training dataset
    output = model(X_train) # pass the correct data through the model, resulting in the output, which is the reconstruction of the data
    loss = criterion(output, X_train) # calculate the error between what the model predicted and what it should have predicted using mean squared error
    optimizer.zero_grad() # reset gradients to prevent accumulation
    loss.backward() # adjust weights and biases (gradients) based on the error, PyTorch handles this automatically
    optimizer.step() # update weights and biases in the model to reduce the loss (error)
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")
        
with torch.no_grad():  # disable gradients as we are not training the model on the test set
    reconstructed = model(X_test)
    reconstruction_error = torch.mean((X_test - reconstructed) ** 2, dim=(1, 2)).numpy()  # calculate the mean squared error between the original
    # and reconstructed data for each test sample

anomalies = np.where(reconstruction_error > (np.mean(reconstruction_error) + 2*np.std(reconstruction_error)))[0]
# np.mean(reconstruction_error), the average reconstruction error for all samples
# np.std(reconstruction_error), the standard deviation of the reconstruction error for all samples

print(anomalies)

for i in anomalies:
    print(f"Anomalous sample {i}: y_noisy[{i}:{i+20}]")
    
for a in anomalies:
    idx_range = range(a, a + 20)
    print(f"Anomaly detected in points: {list(idx_range)}")

# The reconstruction error is low for data similar to the training data and high for anomalies.
# Using the mean and standard deviation, I determine the threshold above which data is considered anomalous.

# The mean reconstruction error indicates how well the model reconstructs the data.
# The standard deviation of the reconstruction error shows how varied the errors are across the entire dataset.
# If the reconstruction error for a specific sample is greater than the mean reconstruction error
# + 2 * the standard deviation of the reconstruction error,
# then that sample is considered an anomaly.
# Therefore, our model treats all arguments belonging to a sample with an anomaly as anomalies.

# if my data is consistent with the normal pattern, autoencoder will be able to reconstruct it and the error will be small/negligible,
# if the input data is atypical, the autoencoder will not be able to reconstruct it well, the error will be large
# then I compare the average reconstruction error for the entire dataset with the reconstruction error for a specific sample,
# if the reconstruction error for a specific sample is greater than the average reconstruction error + 2 * standard deviation of the reconstruction error
# then I treat it as an anomaly


#TODO
# cicha anomalia