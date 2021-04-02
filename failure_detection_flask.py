# Import dependencies
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from flask import Flask, jsonify, request
from torch import nn


# Declare constants
AMPL_THRESH = 2  # Amplitude threshold
BANDS = 32  # Number of frequency bands
FREQ_THRESH = 512  # Frequency threshold
N_FREQ = 12  # Numbers of frequencies
NORMALITY_TRESHOLD = 0.01  # Treshold for making decision about data normality

app = Flask(__name__)


# Define Autoencoder's architecture
class AutoEncoder(nn.Module):
    def __init__(self) -> None:
        super(AutoEncoder, self).__init__()
        # Encoder
        self.conv1 = nn.Conv2d(1, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 32, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # Decoder
        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv5 = nn.Conv2d(64, 1, 3, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.up1(F.relu(self.conv3(x)))
        x = self.up2(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
model.load_state_dict(torch.load("autoencoder_conv_augm_32bands_1channel_data_06032020.pt"))
model.eval()
criterion = nn.MSELoss()


# Define helper functions
def clean_data(data: np.ndarray, n_freq: int = 12,
               ampl_flag: bool = True, ampl_thresh: float = 2,
               freq_flag: bool = True, freq_thresh: float = 512) -> np.ndarray:
    """
    Change amplitudes and frequencies which are 
    more than amplitude/ frequency thresholds to thresholds.

    Parameters
    ----------
    data : Input data with the shape (N, 49), where:
        the 1st dimension is N/2 s time interval,
        the 2nd dimension consists of: 1st value is technical detail, 2-13 - amplitudes, 
        14-25 - frequencies, 26-37 - amplitudes, 38-49 - frequencies.
    
    n_freq : Number of amplitudes (frequencies) for one axis.

    ampl_flag : If this is set to True, amplitudes are reduced to the amplitude threshold.

    ampl_thresh : Amplitude threshold.

    freq_flag : If this is set to True, frequencies are reduced to the frequency threshold.

    freq_thresh : Frequency threshold.

    Returns
    -------
    clean_data : Output data with the shape (N, 48), where from the 2nd dimension technical detail is deleted
              and amplitudes and frequencies are reduced to thresholds.
    """
    
    data_copied = data[:, 1:].copy()  # exclude 1st column
    part1 = data_copied[:, list(range(0, n_freq))]  # amplitude
    part2 = data_copied[:, list(range(n_freq, 2*n_freq))]  # frequency
    part3 = data_copied[:, list(range(2*n_freq, 3*n_freq))]  # amplitude
    part4 = data_copied[:, list(range(3*n_freq, 4*n_freq))]  # frequency

    if ampl_flag:
        part1[part1 > ampl_thresh] = ampl_thresh
        part3[part3 > ampl_thresh] = ampl_thresh
    
    if freq_flag:
        part2[part2 > freq_thresh] = freq_flag
        part4[part4 > freq_thresh] = freq_flag
  
    return np.hstack((part1, part2, part3, part4))


def transform_data(data: np.ndarray, n_freq: int = 12, low_freq: float = 0, 
                   high_freq: float = 512, num_bands: int = 5) -> np.ndarray:
    """
    Transform data into rows with frequency bands.
    Frequency range 0-512 Hz is divided on frequency bands. 
    For each band average amplitude is calculated.

    Parameters
    ----------
    data : Input data with the shape (N, 48), where:
        the 1st dimension is N/2 s time interval,
        the 2nd dimension consists of: 1-12 - amplitudes, 13-24 - frequencies,
        25-36 - amplitudes, 37-48 - frequencies.
    
    n_freq : Number of amplitudes (frequencies) for one axis.

    low_freq : The minimum frequency.

    high_freq : The maximum frequency.

    num_bands : Number of frequency bands.

    Returns
    -------
    transform_data : Output data with the shape (N, num_bands*2), where the 2nd dimension - number of frequency bands multiplied by
                  number of axis. For each frequency band the average ampitude is calculated.
    """

    # Create bins for frequency bands
    bins_freq = np.linspace(low_freq, high_freq, num_bands+1)

    # Transform data in a form of table: rows - time, columns - frequency bands
    rows = []  # transformed raw records
    for record in data:
        row = np.zeros(shape=num_bands*2)

        df_ax1 = pd.DataFrame({
            'ax1_ampl': record[0:n_freq],
            'ax1_freq': record[n_freq:2*n_freq]
        })

        df_ax2 = pd.DataFrame({
            'ax2_ampl': record[2*n_freq:3*n_freq],
            'ax2_freq': record[3*n_freq:4*n_freq]
        })

        df_ax1['band'] = np.digitize(df_ax1['ax1_freq'], bins_freq, right=True)
        df_ax2['band'] = np.digitize(df_ax2['ax2_freq'], bins_freq, right=True)

        df_ax1_grouped = df_ax1.groupby(by=['band']).mean()['ax1_ampl']
        df_ax2_grouped = df_ax2.groupby(by=['band']).mean()['ax2_ampl']

        for idx, value in df_ax1_grouped.items():
            row[idx-1] = value

        for idx, value in df_ax2_grouped.items():
            row[num_bands+idx-1] = value

        rows.append(row)                        
        
    return np.array(rows)[np.newaxis, np.newaxis, :, :]



@app.route("/predict", methods=["POST"])
def predict():
    # Read and transform data
    # Raw_data should have shape (40, 49)
    raw_data = np.array(request.get_json())

    data = clean_data(
        raw_data, n_freq=N_FREQ, 
        ampl_flag=True, ampl_thresh=AMPL_THRESH,
        freq_flag=True, freq_thresh=FREQ_THRESH        
    )
    
    data = transform_data(data, n_freq=N_FREQ, low_freq=0, high_freq=FREQ_THRESH, num_bands=BANDS)

    inputs = torch.tensor(data, dtype=torch.float, device=device)
    loss = criterion(model.forward(inputs), inputs).item()
    result = True
    if loss > NORMALITY_TRESHOLD:
        result = False

    return jsonify({
        "Normality": result,
        "Loss": loss
    })


if __name__ == "__main__":
    app.run()
