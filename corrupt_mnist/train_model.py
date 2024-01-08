import os

import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np

from models.model import SimpleCNN
from data.dataset import mnist

def train(lr = 0.001):
    """Train a model on MNIST."""

    model = SimpleCNN()
    train_set, _ = mnist()
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
    epochs = 15

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for img, label in train_set:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().item())
        
        print(f'Epoch {epoch} training loss: {np.mean(epoch_losses)}')
        losses += epoch_losses

    numfiles = len(os.listdir('./models'))
    torch.save(model, f'./models/trained_model_{numfiles}.pt')

    plt.plot(losses)
    plt.savefig(f'./reports/figures/latest_plot_{numfiles}.png')
    plt.show()

if __name__ == '__main__':
    train()