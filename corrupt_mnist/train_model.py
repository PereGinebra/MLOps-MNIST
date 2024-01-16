import os

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
import hydra
import wandb

from .models.model import SimpleCNN
from .data.dataset import mnist
from .log.my_logger import logger

@hydra.main(config_path='../',config_name='config.yaml',version_base='1.2')
def train(cfg):
    """Train a model on MNIST."""
    wandb.init(config=dict(cfg.hyperparams))
    torch.manual_seed(cfg.hyperparams.torch_seed)

    train_set, _ = mnist()
    train_dl = DataLoader(train_set, batch_size=cfg.hyperparams.batch_size)

    model = SimpleCNN()
    loss_fn = CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.hyperparams.lr)
    epochs = cfg.hyperparams.epochs

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    logger.info(f'Training model on the {str(device)} device')
    #print(f'Training model on the {str(device)} device')

    wandb.watch(model, log_freq=100)
    model.train()
    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for img, label in train_dl:
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.detach().cpu().item())
            wandb.log({'loss':loss.detach().cpu().item()})
        
        logger.info(f'Epoch {epoch} training loss: {np.mean(epoch_losses)}')
        losses += epoch_losses

    numfiles = len(os.listdir('./models'))
    torch.save(model, f'./models/trained_model_{numfiles}.pt')

    plt.plot(losses)
    plt.savefig(f'./reports/figures/latest_plot_{numfiles}.png')

if __name__ == '__main__':
    train()