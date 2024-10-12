import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
import cv2 as cv
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

N = 2000 # number of images

# Get dataset
X0 = np.ones((N,128,44))
X0_mean = np.mean(np.mean(np.mean(X0)))
X0_var = np.var(np.var(np.var(X0)))
for i in range(0,N):
    X0[i,:,:] = cv.cvtColor(cv.imread("dataset_simple_synth/testemusica"+str(i)+".png"),cv.COLOR_BGR2GRAY)
    xxxxxxx = X0[0,:,:]
    X0[i,:,:] = (X0[i,:,:] - X0_mean) / 256
    xxxxxxx = X0[0,:,:]

X = torch.tensor(X0,requires_grad=True)
Y = torch.tensor(np.genfromtxt("dataset_simple_synth/data1.csv", delimiter=',', skip_header=1))

Y = Y[:,1:]
Y = Y[:N,:]



#-----
Y_mean = torch.mean(Y,dim=0)
for i in range(0,2):
    Y[:,i] = Y[:,i] / Y_mean[i]
Y = Y.float()


class MusicDataset(Dataset):

    def __init__(self):
        self.n_samples = N

        self.x_data = X
        self.y_data = Y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = MusicDataset()

batch_size = 20 

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#-----
sr = 44100


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,16,7)
        self.conv2 = torch.nn.Conv2d(16,32,3)
        self.conv3 = torch.nn.Conv2d(32,40,3)
        #self.conv4 = torch.nn.Conv2d(40,80,3)
        #self.conv5 = torch.nn.Conv2d(80,80,3)
        #self.conv6 = torch.nn.Conv2d(80,100,3)
        #self.conv7 = torch.nn.Conv2d(100,100,3)
        #self.conv8 = torch.nn.Conv2d(512,512,3)
        #self.conv9 = torch.nn.Conv2d(512,512,3)
        #self.conv10 = torch.nn.Conv2d(512,512,3)
        #self.conv11 = torch.nn.Conv2d(512,512,3)
        #self.conv12 = torch.nn.Conv2d(512,512,3)
        #self.conv13 = torch.nn.Conv2d(512,512,3)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()
        self.act5 = nn.ReLU()
        self.act6 = nn.ReLU()
        self.act7 = nn.ReLU()
        self.act8 = nn.ReLU()
        self.act9 = nn.ReLU()
        self.act10 = nn.ReLU()


        self.max1 = torch.nn.MaxPool2d(2)
        self.max2 = torch.nn.MaxPool2d(2)
        self.max3 = torch.nn.MaxPool2d(2)
        self.max4 = torch.nn.MaxPool2d(2)

        self.linear1 = torch.nn.Linear(9280,32)
        self.linear2 = torch.nn.Linear(32,12)
        self.linear3 = torch.nn.Linear(12,2)

        self.act14 = nn.ReLU()
        self.act15 = nn.ReLU()



        #self.flatten_layer = torch.nn.Flatten()
    def forward(self, x):
        x = x.float()
        x = x.view(batch_size,1, 128,44)

        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.max1(x)

        x = self.conv3(x)
        x = self.act3(x)
        #x = self.conv4(x)
        #x = self.act4(x)
        x = self.max2(x)

        #x = self.conv5(x)
        #x = self.act5(x)
        #x = self.conv6(x)
        #x = self.act6(x)
        #x = self.conv7(x)
        #x = self.act7(x)
        #x = self.max3(x)

        x = torch.flatten(x,start_dim=1)

        x = self.linear1(x)
        x = self.act14(x)
        x = self.linear2(x)
        x = self.act15(x)
        latent = self.linear3(x)
        

        return latent

model = Model()


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

n_epochs = 10

e = []
for epoch in range(n_epochs):
    for batch_x, batch_y in dataloader:
        Y_pred = model(batch_x)
        loss = loss_fn(Y_pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e.append(loss.item())

    print(f'Época: {epoch}, loss: {loss}')

dataiter = iter(dataloader)
data_test = next(dataiter)
x_test, y_test = data_test
y_pred_test = model(x_test)


plt.plot(e)
plt.show() 


aaaa = 1
ssss = 2
#-----------------------------------------------------------

