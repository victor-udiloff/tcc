
import customtkinter as ctk
import signall
import librosa
import sounddevice as sd
import time
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import pandas as pd
#-----------------------------------------------------------

sr = 44100

def generate(z):
    timee=2
    audio1 = signall.osc(z[0],z[5],timee)
    audio2 = signall.osc(z[1],z[6],timee)
    audio = (audio1+audio2)/2
    if np.max(audio) == 0:
        return audio
    audio = signall.filter_song(audio,z[8],z[7],z[2])
    audio = signall.adrs(audio,z[9],z[10],z[12],z[11])
    if np.max(audio) == 0:
        return audio
    audio = signall.add_distortion(audio,"gamma",z[13])
    audio = signall.add_reverb(audio,z[3])
    return audio

def create_dataset(N):

    z = np.ones((N,14))
    for i in range(0,N):

        aaa = float(np.random.rand(1))
        bbb = float(np.random.rand(1))
        ccc = float(np.random.rand(1))
        ddd = float(np.random.rand(1))
        waveform1 = 0
        waveform2 = 0
        if aaa > 0.333 and aaa < 0.666:
            waveform1 = 1
        if aaa > 0.666:
            waveform1 =2
        if bbb > 0.333 and aaa < 0.666:
            waveform2 = 1
        if bbb > 0.666:
            waveform2 =2
        type = 0
        if ccc > 0.5:
            type = 1
        reverb = 0
        if ddd > 0.5:
            reverb = 1
        

        mix = 50
        f1 = 440
        f2 =100 + round(int(2000*np.random.rand(1)))
        cutoff = 100 + round(int(5000*np.random.rand(1)))
        order = 1 + round(int(10*np.random.rand(1)))
        a = 1 + round(int(500*np.random.rand(1)))
        d = 100 + round(int(500*np.random.rand(1)))
        s = 1 + round(int(800*np.random.rand(1)))
        r = 0 + round(int(40*np.random.rand(1))) 
        amount =  round(int(100*np.random.rand(1)))
        z[i,:]=[waveform1,waveform2,type,reverb,mix,f1,f2,cutoff,order,a,d,s,r,amount]
        audio = generate(z[i,:])
        if  0 < i < 10:
            name = "dataset/testemusica"+str(i)+".wav"
            sf.write(name,audio,44100)
        imagem = librosa.feature.melspectrogram(audio,sr)
        plt.imsave("dataset/testemusica"+str(i)+".png",imagem)

    print(z)
    print(z.shape)
    DF = pd.DataFrame(z)
    
    # save the dataframe as a csv file
    DF.to_csv("dataset/data1.csv")
    print(DF)

def test_data(yp,y):
    y_audio = generate(y)
    yp_audio = generate(yp)
    
    sd.play(y_audio)
    time.sleep(5)
    sd.stop()
    
    sd.play(yp_audio)
    time.sleep(5)
    sd.stop()


def create_simple_synth_dataset(N):

    z = np.random.rand(N,2)
    
    z[:,0] = np.power(z[:,0],4)
    z[:,1] *= 20000
    z[:,1] += 20

    for i in range(0,N):
        x = np.linspace(0,3,int(0.5*sr))
        y  = z[i,0] * np.cos(x*z[i,1])
        if  0 < i < 10:
            name = "dataset_simple_synth/testemusica"+str(i)+".wav"
            #sf.write(name,y,44100)
        imagem = librosa.feature.melspectrogram(y,sr)
        plt.imsave("dataset_simple_synth/testemusica"+str(i)+".png",imagem)


    DF = pd.DataFrame(z)
    
    # save the dataframe as a csv file
    DF.to_csv("dataset_simple_synth/data1.csv")
    print(DF)
#create_dataset(2000)




'''
a20 = np.array([4.8221e+01, 6.5675e-01, 7.5035e-01, 5.1344e-01, 0.0000e+00, 4.8365e+01,
        4.2781e+02, 1.0224e+03, 2.5531e+03, 5.2004e+00, 2.3272e+02, 3.4081e+02,
        4.8438e+02, 1.8445e+01, 4.6524e+01])

y_pred_test = [0.0000, 1.1080, 1.9802, 1.8868, 1.5848, 0.1593, 0.1786, 1.1767, 0.5663,
         0.5445, 0.8463, 1.1387]
y_test = 
'''
#test_data(y_pred_test,y_test)

#sd.play(generate(np.round(a20)),44100)
#time.sleep(5)
#sd.stop()

create_simple_synth_dataset(2000)