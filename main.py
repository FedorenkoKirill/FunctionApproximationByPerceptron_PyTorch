import numpy as np
import math
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt

def f(x):
    return math.sin(x)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(1, 70),
            nn.Tanh(),
            nn.Linear(70, 70),
            nn.Tanh(),
            nn.Linear(70, 1)
        )

    def forward(self, x):
        #x = F.sigmoid(self.lin1(x))
        #x = F.tanh(self.lin2(x))
        #x = F.relu(self.lin3(x))
        return self.layers(x)


if __name__ == "__main__":
    x = np.linspace(-3, 3, 10000).reshape(-1, 1)
    f = np.vectorize(f)
    # вычисляем вектор значений функции
    y = f(x)

    lr = 0.01
    n_epochs = 800
    plot_every = n_epochs / 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train_tensor = torch.from_numpy(x).float().to(device)
    y_train_tensor = torch.from_numpy(y).float().to(device)

    model = Net()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, nesterov=True)

    losses = []

    for epoch in range(n_epochs):
        model.train()
        yhat = model(x_train_tensor)

        loss = loss_fn(y_train_tensor, yhat)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)

        if (epoch % plot_every) == 0:
            plt.scatter(x, y, color='blue', linewidth=2)
            plt.plot(x, yhat.cpu().detach().numpy(), color='red', linewidth=2, antialiased=True)
            plt.show()

    plt.plot(losses)
    plt.show()