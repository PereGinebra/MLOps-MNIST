import torch
from torch.utils.data import TensorDataset, DataLoader

def mnist():
    train_images = torch.load('./data/processed/train_images.pt')
    train_target = torch.load('./data/processed/train_target.pt')
    train_ds = TensorDataset(train_images, train_target)
    train_dl = DataLoader(train_ds, batch_size=64)

    test_images = torch.load('./data/processed/test_images.pt')
    test_target = torch.load('./data/processed/test_target.pt')
    test_ds = TensorDataset(test_images, test_target)
    test_dl = DataLoader(test_ds, batch_size=64)

    return train_dl, test_dl

if __name__ == '__main__':
    pass