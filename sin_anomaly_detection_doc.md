# Time Series Anomaly Detection with Advanced Autoencoders

This project demonstrates a system designed to automatically detect unusual patterns or "anomalies" in time-series data, specifically using the `abs(sin)` function as an example, mocked data. It covers a machine learning technique called Autoencoders, combined with multiple specialized detectors to find various types of abnormal behaviors in a signal.

## 1. What is Anomaly Detection?

You have a sensor to get some data. Most of the time, the data follows a predictable pattern – this is normal, usual behavior. Anomaly detection is the process of finding data points or patterns that are significantly different from this normal behavior. We're looking for different kinds of "weird" behaviors in our `abs(sin)` signal, like sudden jumps, changes in its rhythm, or changes in its strength.

### Why "Unsupervised" Anomaly Detection?

We only have examples of normal, healthy behavior. Unsupervised anomaly detection means our system learns what normal looks like, and then anything that deviates significantly from this learned normal is flagged as an anomaly. So we don't need to use any label, because the entire training set is known as a correct one.

### How Autoencoders Help (The Core Idea)

An **Autoencoder** is a special type of neural network. Think of it like a smart "compressor and decompressor" for data:

1.  **Compression (Encoder)**: It takes an input (e.g., a piece of your signal) and learns to compress it into a much smaller, concise representation (like making a big photo into a tiny thumbnail).
2.  **Decompression (Decoder)**: It then takes this tiny compressed version and tries to decompress it back into something that looks exactly like the original input.

**The Clever Part for Anomaly Detection**:

*   **Training on Normal Data**: We train the Autoencoder *only* on perfectly "normal" data. It becomes very good at compressing and decompressing normal patterns.
*   **Detecting Anomalies**: When we give the trained Autoencoder a *new* piece of data:
    *   If the new data is "normal," the Autoencoder will reconstruct it almost perfectly (low error).
    *   If the new data is "anomalous" (something the Autoencoder has never seen during training), it will struggle to reconstruct it accurately. The difference between the original anomalous input and the Autoencoder's reconstructed output will be very large.

This "reconstruction error" becomes our main indicator of an anomaly. A high error means "this looks strange!"

## 2. Project Concept

Our project takes the `abs(sin)` function data and applies a sophisticated, multi-stage approach to find anomalies:

1.  **Smart Data Preparation (Feature Extraction)**: Instead of just looking at the raw signal points, we first divide the signal into small, overlapping chunks (called "segments"). For each segment, we extract a rich set of "features" or characteristics. These features describe various aspects of the segment, like its overall shape, its rhythm (frequency), its starting point (phase), and its strength (amplitude). This gives the Autoencoder much more information to work with.

2.  **Learning "Normal" (Autoencoder Training)**:
    *   We take a perfectly clean, normal `abs(sin)` signal.
    *   We extract the advanced features from *only* this normal signal.
    *   We then train our `AdvancedAutoencoder` model using these normal features. The Autoencoder learns the typical relationships and patterns within these features. It learns to compress and decompress them perfectly when they are "normal."

3.  **Finding the "Strange" (Anomaly Detection with Ensemble)**:
    *   When we receive a new signal that might contain anomalies, we apply the *exact same* feature extraction process to it.
    *   We feed these new features into our *already trained* `AdvancedAutoencoder`.
    *   The Autoencoder tries to reconstruct the features. If it reconstructs them poorly, it's a sign of an anomaly.
    *   **Ensemble Detection**: To make our detection even more robust and capable of catching different *types* of anomalies, we don't just rely on the Autoencoder's reconstruction error. We use an "ensemble" of detectors:
        *   One detector checks the overall reconstruction error.
        *   Other specialized detectors look for specific changes in frequency, phase, or amplitude by comparing the noisy signal segments to what a clean segment would look like.
    *   If *any* of these detectors flags a segment as suspicious, we consider that segment and the corresponding part of the original signal to contain an anomaly.

This combined approach allows our system to be sensitive to a wide range of unusual patterns in the signal.

## 3. File-by-File Breakdown

Let's dive into the details of each Python file in the project.

### `detection_components.py`

This file contains the core building blocks for our anomaly detection system: the feature extraction logic, the Autoencoder neural network, and the multi-detector ensemble.

#### `extract_advanced_features(data, seq_len=20, overlap=0.5)` function

*   **Purpose**: This is a crucial function that takes a raw time-series signal (`data`) and transforms it into a set of informative "features" for each small piece of the signal. These features help the Autoencoder understand the signal's characteristics much better than just looking at raw values.
*   **How it works**:
    1.  **Segmenting the Data**: It breaks the long `data` signal into smaller, overlapping `segments` (each `seq_len` samples long, e.g., 20 samples). `overlap` ensures that we don't miss anomalies that might fall on segment boundaries.
    2.  **Calculating Features for Each Segment**: For each segment, it calculates a wide variety of characteristics:
        *   **Time Domain Features**:
            *   `segment`: The raw signal values in the segment.
            *   `first_derivative`: How much the signal changes from one point to the next. Helps detect sudden spikes or drops.
            *   `second_derivative`: How quickly the change itself is changing (acceleration). Helps detect sharp turns or rapid changes.
        *   **Frequency Domain Features (using Fast Fourier Transform - FFT)**:
            *   `fft_values`: Shows which frequencies are present in the segment and how strong they are.
            *   `dominant_freq`, `dominant_magnitude`: The most prominent frequency and its strength. Helps identify changes in the signal's rhythm.
            *   `freq_spread`: How spread out the energy is across different frequencies. A high spread can indicate a noisy or changing frequency pattern.
            *   `spectral_centroid`: The "center of mass" of the frequencies. A shift here indicates a change in the dominant frequency components.
        *   **Phase Features**:
            *   `fft_phase`: The "starting point" or alignment of different frequency components.
            *   `phase_mean`, `phase_std`: Average phase and how much it varies. Helps detect `phase shifts` (where the signal is shifted horizontally).
        *   **Amplitude Features (using Hilbert Transform)**:
            *   `amplitude_envelope`: The "outline" or overall strength of the signal over time.
            *   `envelope_mean`, `envelope_std`: Average and variability of the signal's strength. Helps detect `attenuator` anomalies (where the signal becomes much weaker).
        *   **Statistical Features**:
            *   `mean`: The average value of the segment.
            *   `std`: Standard deviation, showing how much the values in the segment vary. High `std` means a lot of change.
            *   `skew`: Measures the asymmetry of the segment's distribution.
            *   `kurtosis`: Measures the "tailedness" of the distribution; high kurtosis can indicate more extreme values or outliers.
        *   **Wavelet Features**:
            *   `smoothed`: A smoothed version of the signal, removing sharp details.
            *   `detail`: The difference between the original and smoothed signal, highlighting sharp changes and anomalies.
            *   `detail_energy`: The strength of these sharp details.
    3.  **Combining and Normalizing**: All these individual features for a segment are combined into one long "feature vector." Finally, these feature vectors are `normalized` using `StandardScaler`. This makes sure all features have a similar scale (e.g., mean of 0, standard deviation of 1), which is important for the Autoencoder to learn effectively.
*   **Output**: Returns the normalized feature vectors, the `StandardScaler` object (which remembers how the data was scaled), and the starting index (`indices`) of each segment in the original signal.

#### `AdvancedAutoencoder(nn.Module)` class

*   **Purpose**: This class defines the structure of our Autoencoder neural network, which is designed to learn and reconstruct the "normal" patterns of our advanced features.
*   **`__init__(self, input_size)` method**:
    *   This is where the Autoencoder's "architecture" (its layers and connections) is set up.
    *   It has an `encoder` part and a `decoder` part.
    *   **`encoder`**: This part compresses the input features. It takes the `input_size` (the length of our feature vector) and reduces it down to a `bottleneck_size` (e.g., 16).
        *   It uses `nn.Linear` layers for computations.
        *   `LeakyReLU(0.2)`: An "activation function" that helps the network learn complex, non-linear relationships. It's a slightly "softer" version of `ReLU`.
        *   `Dropout(0.2)`: This is a technique to prevent "overfitting" (where the model memorizes the training data too well and performs poorly on new data). During training, it randomly turns off 20% of the neurons, forcing the network to learn more robust and general patterns.
    *   **`decoder`**: This part takes the compressed information from the `bottleneck_size` and tries to expand it back to the original `input_size`. It uses similar `nn.Linear` and `LeakyReLU` layers.
*   **`forward(self, x)` method**:
    *   This method defines the "flow" of data through the Autoencoder. When you give the Autoencoder an input `x`, it first passes `x` through the `encoder` to get a compressed representation, and then passes this compressed representation through the `decoder` to get the reconstructed output.

#### `AnomalyDetectorEnsemble` class

*   **Purpose**: This class brings together multiple anomaly detection strategies. Instead of relying on just one way to spot anomalies, it uses several "detectors" in combination. This makes the system more robust and capable of finding different *types* of anomalies that a single detector might miss.
*   **`__init__(self, X_test, reconstructed, test_indices, seq_len)` method**:
    *   Initializes the ensemble with the original features (`X_test`), the features reconstructed by the Autoencoder (`reconstructed`), the starting indices of the segments, and the segment length. It also creates a `set()` called `anomalies` to store unique indices of detected anomalies.
*   **`reconstruction_error_detector(self, sensitivity=2.5)` method**:
    *   **Purpose**: This is the primary detection method based on the Autoencoder's performance.
    *   **How it works**: It calculates the "Mean Squared Error" (`mse`) between the original features (`X_test`) and the `reconstructed` features. A higher `mse` means the Autoencoder struggled more, indicating a potential anomaly. It converts these errors into "Z-scores" (how many standard deviations away from the average error) and flags segments where the Z-score exceeds a `sensitivity` threshold (e.g., 2.5 means 2.5 standard deviations above average).
*   **`frequency_detector(self, y_clean, y_noisy, window_size=20, sensitivity=3.0)` method**:
    *   **Purpose**: Specifically designed to catch anomalies where the signal's *frequency* pattern changes (e.g., `3x frequency` anomaly).
    *   **How it works**: It extracts dominant frequencies from both the `clean` reference signal and the `noisy` signal in small windows. If the frequency of a noisy segment is significantly different from its clean counterpart, it's flagged as a frequency anomaly.
*   **`phase_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0)` method**:
    *   **Purpose**: Aims to detect `phase shift` anomalies, where the signal is shifted horizontally compared to its normal pattern.
    *   **How it works**: It calculates and compares the phase information (using Hilbert transform) of segments from the `clean` signal and the `noisy` signal. A large difference suggests a phase anomaly.
*   **`amplitude_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0)` method**:
    *   **Purpose**: Catches anomalies related to changes in the signal's *amplitude* or overall strength (e.g., `attenuator` anomaly).
    *   **How it works**: It compares the `std` (standard deviation, indicating variability) or amplitude envelope of segments from the `clean` signal and the `noisy` signal. A significant change in amplitude characteristics flags an anomaly.
*   **`run_all_detectors(self, y_clean, y_noisy)` method**:
    *   **Purpose**: Executes all the individual detection methods (`reconstruction_error_detector`, `frequency_detector`, `phase_detector`, `amplitude_detector`).
    *   **Combining Results**: It collects all the anomaly indices found by any of the detectors into a single set, ensuring that each anomaly is counted only once.
    *   **Output**: Returns a list of unique segment indices identified as anomalous, along with the raw metrics (errors, differences, ratios) from each detector for further analysis.

### `train_autoencoder.py`

This script is responsible for the crucial first step: teaching the `AdvancedAutoencoder` what "normal" data looks like.

*   **`main()` function**:
    1.  **Generate Clean Data**: It starts by creating a perfectly `y_clean` `abs(sin)` signal. This signal represents the "normal" behavior we want our Autoencoder to learn.
    2.  **Prepare Training Features**: It calls `extract_advanced_features` (from `detection_components.py`) to process this `y_clean` signal into `X_train_features`. This ensures the Autoencoder learns from the same type of features it will see during detection.
    3.  **Initialize Autoencoder**: An `AdvancedAutoencoder` model is created.
    4.  **Define Loss and Optimizer**:
        *   `criterion = nn.MSELoss()`: This is the "loss function." It measures how different the Autoencoder's output is from the original input. The goal during training is to make this `loss` as small as possible.
        *   `optimizer = torch.optim.Adam(...)`: This is the "optimizer." It's an algorithm that helps the Autoencoder adjust its internal settings (weights) based on the `loss` to improve its performance.
    5.  **Training Loop**:
        *   The code then enters a loop (`for epoch in range(epochs)`), where the Autoencoder repeatedly processes the `X_train` data.
        *   In each `epoch`:
            *   `output = model(X_train)`: The Autoencoder tries to reconstruct the input features.
            *   `loss = criterion(output, X_train)`: The `loss` is calculated, telling us how well the reconstruction was.
            *   `optimizer.zero_grad()`: Clears previous calculations.
            *   `loss.backward()`: Calculates how much each weight in the network contributed to the `loss`.
            *   `torch.nn.utils.clip_grad_norm_`: This is a safeguard that prevents extremely large weight updates, which can destabilize training.
            *   `optimizer.step()`: The Autoencoder adjusts its weights to reduce the `loss`.
        *   **Progress Monitoring**: Every 100 epochs, it prints the current `loss` to show training progress.
        *   **Early Stopping**: If the `loss` becomes very small (less than `0.0001`), the training stops early, as the model has likely learned the normal patterns sufficiently well.
    6.  **Save Trained Model**: After training, `torch.save(model.state_dict(), 'advanced_autoencoder.pth')` saves the learned knowledge of the Autoencoder into a file named `advanced_autoencoder.pth`. This file can then be loaded later for actual anomaly detection without needing to retrain the model.

### `detection_autoencoder.py`

This script simulates the "real-time" anomaly detection process using the pre-trained Autoencoder model.

*   **Simulated Noisy Data**:
    *   It first creates a `y_clean` `abs(sin)` signal, just like in training.
    *   Then, it generates several `y_noisy` versions of this signal (`y_noisy`, `y_noisy1`, `y_noisy2`, `y_noisy3`). Each of these `y_noisy` signals has different *known* anomalies introduced at specific positions (e.g., common spikes, 3x frequency changes, phase shifts, amplitude changes). This allows us to test if our detection system can find them.
    *   These noisy signals are collected in the `detector` list.
*   **`real_time_anomaly_detection(y_detected)` function**:
    *   **Purpose**: This function takes a single (potentially noisy) signal (`y_detected`) and runs the full anomaly detection pipeline on it.
    *   **Prepare Features**: It calls `extract_advanced_features` to get the feature vectors for the `y_detected` signal.
    *   **Load Pre-trained Model**: `model.load_state_dict(torch.load('advanced_autoencoder.pth'))` loads the Autoencoder's learned knowledge from the file saved by `train_autoencoder.py`.
    *   `model.eval()`: This command tells the model to switch to "evaluation mode." In this mode, features like `Dropout` (which are only used during training) are turned off, ensuring consistent and predictable results.
    *   **Reconstruct Features**: `with torch.no_grad(): reconstructed = model(X_test)` feeds the new features into the loaded Autoencoder. `torch.no_grad()` means we are not training the model, so we don't need to track information for updating weights, making the process faster.
    *   **Run Ensemble Detectors**: It creates an `AnomalyDetectorEnsemble` object and calls its `run_all_detectors` method. This runs all the individual detectors (reconstruction error, frequency, phase, amplitude) on the `y_detected` signal, comparing it against the `y_clean` baseline where needed.
    *   **Display Results**: It prints the starting and ending positions of the segments identified as anomalous by the ensemble.
*   **`main()` function**:
    *   This function iterates through each of the simulated `y_noisy` signals in the `detector` list.
    *   For each noisy signal, it calls `real_time_anomaly_detection()` to perform the anomaly detection.
    *   It prints separators (`---`) between the results for different noisy signals for clarity.

## 4. How to Run the Project

Follow these steps to set up and run the anomaly detection project:

### 1. Prerequisites

Make sure you have Python installed (version 3.7 or newer is recommended). You will also need `pip`, Python's package installer, which usually comes with Python.

### 2. Install Required Libraries

Open your terminal or command prompt and run the following command to install all the necessary Python libraries:

```bash
pip install torch numpy matplotlib scikit-learn scipy
