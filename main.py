import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import time
#Задаем функцию, которую и будем аппроксимировать нейронной сетью
def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)


class ApproximationFunctionDataset(Dataset):
    def __init__(self, values, labels):
        self.values = values
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.values[idx]
        sample = {'x': data, 'y': label}
        return sample

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(1, 70)
        self.lin2 = nn.Linear(70, 70)
        self.lin3 = nn.Linear(70, 1)

    def forward(self, x):
        x = torch.tanh(self.lin1(x))
        x = torch.tanh(self.lin2(x))
        x = self.lin3(x)
        return x

if __name__ == "__main__":
    start_time = time.time()
    lr = 0.01
    n_epochs = 400
    plot_every = n_epochs / 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    x = np.linspace(-5, 5, 8000).reshape(-1, 1)
    f = np.vectorize(f)
    y = f(x)

    x_train_tensor = torch.from_numpy(x).float().to(device)
    y_train_tensor = torch.from_numpy(y).float().to(device)

    training_dataset = ApproximationFunctionDataset(x_train_tensor, y_train_tensor)
    training_dataloader = DataLoader(training_dataset, batch_size=512, shuffle=True)

    model = NeuralNetwork()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, nesterov=True)

    losses = []
    for epoch in range(n_epochs):
        for i, data in enumerate(training_dataloader):
            inputs, labels = data.values()

            model.train()
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        if (epoch % plot_every) == 0:
            plt.scatter(x, y, color='blue', linewidth=2)
            plt.plot(x, model(x_train_tensor).detach().numpy(), color='red', linewidth=2, antialiased=True)
            plt.show()

    plt.plot(losses)
    plt.show()
    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))









    """
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
    plt.show()"""