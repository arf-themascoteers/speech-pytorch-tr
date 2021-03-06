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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)




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


# Create training and testing split of the data. We do not use validation in this tutorial.
train_set = SubsetSC("training")
test_set = SubsetSC("testing")

waveform, sample_rate, label, speaker_id, utterance_number = train_set[0]

print("Shape of waveform: {}".format(waveform.size()))
print("Sample rate of waveform: {}".format(sample_rate))

plt.plot(waveform.t().numpy());
plt.show()

waveform_first, *_ = train_set[0]
print(sample_rate)
print(waveform_first.numpy())
wavedata = waveform_first.numpy()
#playsound(waveform_first.numpy())
# wave_read = wave.open("welcome.wav", 'rb')
# wave_obj = sa.WaveObject.from_wave_read(wave_read)
# play_obj = wave_obj.play()
# play_obj.wait_done()

sa.play_buffer(wavedata,1,4,sample_rate)

path = 'SpeechCommands/speech_commands_v0.02/backward/0a2b400e_nohash_0.wav'
info = WavInfoReader(path)
print(info)