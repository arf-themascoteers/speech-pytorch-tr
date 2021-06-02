import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchaudio.datasets import SPEECHCOMMANDS
import os
from playsound import playsound
import simpleaudio as sa
import wave
from wavinfo import WavInfoReader


class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")
validation_set = SubsetSC("validation")
dev_set = validation_set


# for train_data in train_set:
#     waveform, sample_rate, label, speaker_id, utterance_number = train_data
#     print("Shape of waveform: {}".format(waveform.size()))
#     print("Sample Rate: {}".format(sample_rate))

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(16000, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def forward(self, x):
        x = self.net(x)
        return F.log_softmax(x, dim=1)

