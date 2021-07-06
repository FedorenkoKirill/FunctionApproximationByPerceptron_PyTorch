import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt

#Задаем функцию, которую и будем аппроксимировать нейронной сетью
def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)

#Создаем модель нейронной сети
class NeuralNetwork(nn.Module):
    """
    One input layer
    One hidden layer
    One output layer
    """
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.lin1 = nn.Linear(1, 70)
        self.lin2 = nn.Linear(70, 70)
        self.lin3 = nn.Linear(70, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = self.lin3(x)
        return x

if __name__ == "__main__":
    x = np.linspace(-5, 3, 4000).reshape(-1, 1)
    f = np.vectorize(f)
    # вычисляем вектор значений функции
    y = f(x)

    lr = 0.01
    n_epochs = 10000
    plot_every = n_epochs / 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x_train_tensor = torch.from_numpy(x).float().to(device)
    y_train_tensor = torch.from_numpy(y).float().to(device)

    model = NeuralNetwork()
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