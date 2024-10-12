import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,Dataset


N = 20000 # number of samples

A = 2 
w = 15
th = 0.2

# Get dataset
X0 = np.random.rand(N)
Y0 = A* np.cos(w*X0 + th)
X = torch.tensor(X0,requires_grad=True)
Y = torch.tensor(Y0,requires_grad=True)
X = X.float()
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

batch_size = 50 

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Model(nn.Module):
    def __init__(self):
        super().__init__()



        self.linear1 = torch.nn.Linear(1,8)
        self.linear2 = torch.nn.Linear(8,10)
        self.linear3 = torch.nn.Linear(10,14)
        self.linear4 = torch.nn.Linear(14,8)
        self.linear5 = torch.nn.Linear(8,1)

        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.act3 = nn.ReLU()
        self.act4 = nn.ReLU()



        #self.flatten_layer = torch.nn.Flatten()
    def forward(self, x):
        x = x.float()
        x = x.view(batch_size,1)

        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.act3(x)
        x = self.linear4(x)
        x = self.act4(x)
        latent = self.linear5(x)
        latent = latent.view(batch_size)
        return latent

model = Model()


loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

x0_test = x_test.detach().numpy()
y0_test = y_test.detach().numpy()
y0_pred_test = y_pred_test.detach().numpy()

print("x test:",x_test)
print("y test:",y_test)
print("y pred test:",y_pred_test)

#plt.plot(e)
plt.plot(x0_test,y0_test,'o')
plt.plot(x0_test,y0_pred_test,'o')
plt.show() 


