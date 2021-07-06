import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

#Задаем функцию, которую и будем аппроксимировать нейронной сетью
def f(x):
    return x * np.sin(x * 2 * np.pi) if x < 0 else -x * np.sin(x * np.pi) + np.exp(x / 2) - np.exp(0)


class ApproximationFunctionDataset(Dataset):
    def __init__(self, start_point, end_point, count_points):
        self.x = np.linspace(start_point, end_point, count_points).reshape(-1, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.x[idx, 0]
        y = f(x)
        sample = {'x': x, 'y': y}

        return sample

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
        x = x.view(1, -1)
        x = self.lin3(x)
        return x

if __name__ == "__main__":
    """
    x = np.linspace(-5, 3, 4000).reshape(-1, 1)
    f = np.vectorize(f)
    # вычисляем вектор значений функции
    y = f(x)
    """
    training_dataset = ApproximationFunctionDataset(start_point=-3, end_point=3, count_points=2000)
    training_dataloader = DataLoader(training_dataset, batch_size=64, shuffle=True)

    lr = 0.01
    n_epochs = 10000
    plot_every = n_epochs / 5

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    #x_train_tensor = torch.from_numpy(x).float().to(device)
    #y_train_tensor = torch.from_numpy(y).float().to(device)

    model = NeuralNetwork()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=lr, nesterov=True)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(training_dataloader):
            model.train()
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data.values()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            exit()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')









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