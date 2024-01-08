import sys
import pickle

import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    model.to(torch.device('cpu'))
    return torch.cat([model(batch).argmax(dim=-1) for batch in dataloader], 0)

class PredictionDataset(Dataset):
    def __init__(self, data):
        self.data = data if data.ndim > 2 else data.unsqueeze(dim=0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

if __name__ == '__main__':
    if len(sys.argv) == 3:
        model = torch.load(sys.argv[1])
        datafile = sys.argv[2]
        if '.png' in datafile:
            data = np.array(Image.open(datafile))
        elif '.npy' in datafile:
            data = np.load(datafile)
        elif '.pt' in datafile:
            data = torch.load(datafile)
        else:
            data = pickle.load(datafile)
        preds = predict(model, DataLoader(PredictionDataset(data), batch_size=64))
        print(f'Predictions: {preds.tolist()}')
    else:
        print('Incorrect usage of script arguments')