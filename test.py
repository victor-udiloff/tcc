import librosa
import numpy as np
import sounddevice as sd
import time
import matplotlib.pyplot as plt
import math

audio, sr = librosa.load("naturesounds.mp3")
#audio, sr = librosa.load("../audio/snare.wav")

def repeat(song,l,o):
    #5s
    granular = np.zeros(sr* 10) + 0.0001
    grain = song[o:o+l]
    k = math.floor(granular.shape[0]/l)
    for i in range(0,k):
        if ((1+i)*l) < granular.shape[0]:
            granular[(i*l):((1+i)*l)] = grain
    return granular

#plt.plot(audio)
#plt.plot(repeat(audio,40000,50000))
#plt.show()


sd.play(repeat(audio,300,60000))
time.sleep(5)
sd.stop()