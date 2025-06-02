# Time series anomaly detection with advanced autoencoders

This project demonstrates a system designed to automatically detect unusual patterns known as anomalies in time-series data, specifically using the `abs(sin)` function as an example, mocked data. It covers a machine learning technique called Autoencoders, combined with multiple specialized detectors to find various types of abnormal behaviors in a signal.

## 1. What is anomaly detection?

You have a sensor to get some data. Most of the time, the data follows a predictable pattern – this is normal, usual behavior. Anomaly detection is the process of finding data points or patterns that are significantly different from this normal behavior. We're looking for various kinds of weird behaviors in our `abs(sin)` signal, like sudden jumps, changes in its rhythm, or changes in its strength of the signal.

### Why do we use unsupervised anomaly detection?

We only have examples of normal, healthy behavior. Unsupervised anomaly detection means our system learns what normal looks like, and then anything that deviates significantly from this learned normal is flagged as an anomaly. So we don't need to use any label, because the entire training set is known as a correct one.

### The core ideas of an autoencoder.

An autoencoder is a special type of neural network. It works like a compressor and decompressor for data:

1.  Compression - Encoder: It takes an input (a piece of the signal) and learns to compress it into a much smaller, concise representation (like making a big photo into a tiny and poor pixel representation).
2.  Decompression - Decoder: It then takes this tiny compressed version and tries to decompress it back into something that looks exactly like the original input. So it has to know how to capture it, and can detect some deviations, comparing the recreated signal with the original one.
3. Training on Normal Data: The Autoencoder is trained only on perfect, usual data, it becomes very good at compressing and decompressing normal patterns.
4. Detecting Anomalies - we give the trained Autoencoder a new piece of data:
    *   If the new data is "normal," the Autoencoder will reconstruct it almost perfectly (low error).
    *   If the new data is "anomalous" (something the Autoencoder has never seen during training), it will struggle to reconstruct it accurately. The difference between the original anomalous input and the Autoencoder's reconstructed output will be very large, so we can mark that part of the signal as an anomalous one.
5.  Reconstruction error is our main indicator of an anomaly. A high error means an anomaly.

## 2. Project Concept

Our project takes the abs(sin) function data and applies a sophisticated, multi-stage approach to find anomalies:

1.  Data Preparation (Feature Extraction): We first divide the signal into small, overlapping chunks (segments). For each segment, we extract a rich set of features, that precisely describes the signal, like its overall shape, frequency, phase, and amplitude. This gives the Autoencoder much more information to work with, so the detection is more accurate.

2.  The autoencoder pipeline is taking a perfectly clean, normal abs(sin) signal, extracting the advanced features from only this normal signal, training our AdvancedAutoencoder model using these normal features. So the output of that process is the autoencoder learned the typical relationships and patterns within these features. It learns to compress and decompress them perfectly when they are normal.

3.  Anomaly Detection with Ensemble - we receive a new signal that might contain anomalies, we apply the exact same feature extraction process to it. We feed these new features into our already trained AdvancedAutoencoder, it tries to reconstruct the features. If the reconstruction goes poorly, it's a sign of an anomaly. To make our detection even more capable of catching different types of anomalies, we don't just rely on the Autoencoder's reconstruction error. We use an ensemble of detectors (one detector checks the overall reconstruction error, other specialized detectors look for specific changes in frequency, phase, or amplitude by comparing the noisy signal segments to what a clean segment would look like, if any of these detectors flags a segment as suspicious, we consider that segment to contain an anomaly). This approach allows our system to be sensitive to a wide range of unusual patterns in the signal.

## 3. File-by-File Breakdown

### detection_components.py

Blocks and logic for our anomaly detection system.

#### extract_advanced_features(data, seq_len=20, overlap=0.5) function

*   Purpose: It takes a raw time-series signal and transforms it into a set of informative features for each small piece of the signal. The autoencoder understand the signal characteristic much better, as it is explained above.
*   **How it works**:
    1.  Segmenting the Data: It breaks the long data into smaller, overlapping segments (seq_len samples long, 20 samples). Overlap ensures that we don't miss anomalies that might fall on segment boundaries.
    2.  Calculating Features for Each Segment - a wide variety of characteristics:
        *   **Time Domain Features**:
            *   segment: The raw signal values.
            *   first_derivative: How much the signal changes from one point to the next. Helps detect sudden spikes or drops.
            *   second_derivative: How quickly the change itself is changing (acceleration). Helps detect sharp turns or rapid changes.
        *   **Frequency Domain Features (Fourier Transform - FFT)**:
            *   fft_values: Shows which frequencies are present in the segment and how strong they are.
            *   dominant_freq, dominant_magnitude: Helps identify changes in the signal's rhythm.
            *   freq_spread: A high spread can indicate a noisy or changing frequency pattern.
            *   spectral_centroid: A shift here indicates a change in the dominant frequency components.
        *   **Phase Features**:
            *   fft_phase: The starting point or alignment of different frequency components.
            *   phase_mean, phase_std: Average phase and how much it varies.
        *   **Amplitude Features (Hilbert Transform)**:
            *   amplitude_envelope: The outline or overall strength of the signal over time.
            *   envelope_mean, envelope_std: Helps detect attenuator anomalies (where the signal becomes much weaker).
        *   **Statistical Features**:
            *   mean: The average value of the segment.
            *   std: Standard deviation, showing how much the values in the segment vary. High std means a lot of change.
            *   skew: Measures the asymmetry of the segment's distribution.
            *   kurtosis: High kurtosis can indicate more extreme values or outliers.
        *   **Wavelet Features**:
            *   smoothed: A smoothed version of the signal, removing sharp details.
            *   detail: The difference between the original and smoothed signal, highlighting sharp changes and anomalies.
            *   detail_energy: The strength of these sharp details.
    3.  Combining and Normalizing: All these individual features for a segment are combined into one long feature vector. Vectors are normalized using StandardScaler. This makes sure all features have a similar scale, which is important for the Autoencoder to learn effectively.
So output returns the normalized feature vectors, and the starting index of each segment in the original signal.

#### class AdvancedAutoencoder(nn.Module)

*   Purpose: This class defines the structure of our Autoencoder neural network, to learn and reconstruct the normal patterns.
*   **__init__(self, input_size) method**:
    *   The layers and connections to set up the Autoencoder.
    *   Contains encoder and a decoder part.
    *   **encoder compresses the input features. It takes the input_size, that is the length of our feature vector and reduces it down to a bottleneck_size.
        *   It uses nn.Linear layers for computations.
        *   LeakyReLU(0.2): An activation function that helps the network learn complex, non-linear relationships. The alternative, softer version of ReLu.
        *   Dropout(0.2): This is a technique to prevent overfitting (overfitting is the negative behaviour, model memorizes the training data too well and performs poorly on new data). During training, it randomly turns off 20% of the neurons, forcing the network to learn more robust and general patterns.
    *   **decoder**: This part takes the compressed information from the bottleneck_size and tries to expand it back to the original input_size.
*   **forward(self, x) method**:
    *   Defines the flow of data through the Autoencoder. First an input, then encoder - to get a compressed representation, then decoder - to get the reconstructed output.

#### class AnomalyDetectorEnsemble

*   **Purpose**: Bringing together multiple anomaly detection strategies by its combination, so the system is able to detect more precisely different type of anomalies also the silient ones.
*   **__init__(self, X_test, reconstructed, test_indices, seq_len) method**:
    *   Initializes the ensemble with the original features, the features reconstructed by the Autoencoder, the starting indices of the segments, and the segment length. The set() called anomalies to store unique indices of detected anomalies.
*   **reconstruction_error_detector(self, sensitivity=2.5) method**:
    *   **Purpose**: The primary detection method based on the Autoencoder's performance. It calculates the Mean Squared Error (mse) between the original features and the reconstructed ones. A higher mse means the Autoencoder struggled more, indicating a potential anomaly. It converts these errors into Z-scores (how many standard deviations away from the average error) and flags segments where the Z-score exceeds a sensitivity threshold (2.5 is from 2.5 standard deviations above average).
*   **frequency_detector(self, y_clean, y_noisy, window_size=20, sensitivity=3.0) method**:
    *   **Purpose**: Catching anomalies where the signal's frequency pattern changes.
    *   It extracts dominant frequencies from both the clean reference signal and the noisy signal in small windows. If the frequency of a noisy segment is much different from its clean, it's flagged as a frequency anomaly.
*   **phase_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0) method**:
    *   **Purpose**: Detecting phase shift anomalies, where the signal is shifted horizontally compared to its normal pattern.
    *   It calculates and compares the phase information (using Hilbert transform) of segments from the clean signal and the nois signal. A large difference - phase anomaly.
*   **amplitude_detector(self, y_clean, y_noisy, window_size=20, sensitivity=2.0) method**:
    *   **Purpose**: detect anomalies related to changes in the amplitude or strength (like attenuator anomaly).
    *   It compares the std - standard deviation, indicating variability) or amplitude envelope of segments from the clean signal and the noisy signal. A significant change in amplitude characteristics flags an anomaly.
*   **run_all_detectors(self, y_clean, y_noisy)**:
    *   Purpose: Executes all the detection methods mentioned aboved.
    *   It collects all the anomaly indices into a single set, ensuring that each anomaly is counted only once.
    *   Returns a list of unique segment indices identified as anomalous, along with the raw metrics (errors, differences, ratios) from each detector for further analysis.

### train_autoencoder.py

This script is responsible for teaching the AdvancedAutoencoder what normal data looks like.

*   **main() function**:
    1.  **Generate Clean Data**: It starts by creating a perfectly y_clean abs(sin) signal. This signal represents the normal behavior we want our Autoencoder to learn.
    2.  **Prepare Training Features**: calls extract_advanced_features to process y_clean signal into X_train_features, 
 ensures the Autoencoder learns from the same type of features it will see during detection.
    3.  **Initialize Autoencoder**: An AdvancedAutoencoder` model.
    4.  **Define Loss and Optimizer**:
        *   criterion = nn.MSELoss(): It measures how different the Autoencoder's output is from the original input. The goal during training is to make this loss as small as possible.
        *   optimizer = torch.optim.Adam(...) It's an algorithm that helps the Autoencoder adjust its internal settings (weights) based on the loss to improve its performance.
    5.  **Training Loop**:
        *   The code enters a loop (for epoch in range(epochs)), where the Autoencoder repeatedly processes the train data.
        *   In each epoch:
            *   output = model(X_train): The Autoencoder works and tries to reconstruct the input features.
            *   loss = criterion(output, X_train): The loss is calculated, telling us how well the reconstruction was.
            *   optimizer.zero_grad(): Clears previous calculations and gets ready for next ones.
            *   loss.backward(): Calculates how much each weight in the network contributed to the loss.
            *   torch.nn.utils.clip_grad_norm_: This is for the safe reasons that prevents extremely large weight updates, which can destabilize training.
            *   optimizer.step(): The Autoencoder adjusts its weights to reduce the loss.
        *   Every 100 epochs, it prints the current loss to show training progress.
        *   If the loss becomes very small (less than 0.0001), the training stops early, as the model has likely learned the normal patterns.
    6.  **Save Trained Model**: After training, `torch.save(model.state_dict(), 'advanced_autoencoder.pth')` saves the learned knowledge of the Autoencoder into a file named `advanced_autoencoder.pth`. This file can then be loaded later for actual anomaly detection without needing to retrain the model.

### detection_autoencoder.py

Mechanizm for imitating real-time anomaly detection process using the pre-trained Autoencoder model.

*   **Simulated Noisy Data**:
    *   Firstly, creates a y_clean abs(sin) signal, for later added the anomalies.
    *   Then, it generates several y_noisy versions of this signal. Each of these signals has different anomalies introduced at specific positions.
    *   Noisy signals are collected in the detector list.
*   **real_time_anomaly_detection(y_detected) function**:
    *   **Purpose**: This function takes a single, potentially noisy signal and runs the full anomaly detection pipeline on it.
    *   It calls extract_advanced_features to get the feature vectors for the y_detected signal.
    *   model.load_state_dict(torch.load('advanced_autoencoder.pth')) loads the Autoencoder's learned knowledge from the file saved by train_autoencoder.py.
    *   model.eval(): switches to evaluation mode - features like Dropout, which are only used during training are turned off, ensuring consistent and predictable results.
    *   with torch.no_grad(): reconstructed = model(X_test)feeds the new features into the loaded Autoencoder. torch.no_grad() means we are not training the model, so we don't need to track information for updating weights, it makes the process faster.
    *   Run Ensemble Detectors: It creates an AnomalyDetectorEnsemble object and calls its run_all_detectors method. This runs all the individual detectors (reconstruction error, frequency, phase, amplitude) on the y_detected signal, comparing it against the y_clean baseline where needed.
    *   Display Results: It prints the starting and ending positions of the segments identified as anomalous by the ensemble.
*   **main()**:
    *   iterates through each of the y_noisy signals in the detector list.
    *   For each noisy signal, it calls real_time_anomaly_detection() to perform the anomaly detection.

## 4. How to Run the Project

### 1. Prerequisites

    * Python installed (version 3.12 or newer).
    * pip (version 25.1.1, lower might not works).

### 2. Install Required Libraries
    * python3 -m venv venv_name (to create the virtual environment)
    * source venv_name/bin/activate
    * pip install -r requirements.txt
    * python3 train_autoencoder.py (for running the learning process)
    * python3 detection_autoencoder.py (for running anomaly detection)