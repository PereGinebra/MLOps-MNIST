import torch
import shutil

def normalize(images):
    mean = images.mean()
    std = images.std()

    return (images - mean) / std

if __name__ == '__main__':
    train_images = torch.tensor([])
    train_targets = torch.tensor([], dtype=int)
    for i in range(6):
        imgs = torch.load(f'./data/raw/train_images_{i}.pt')
        train_images = torch.cat([train_images, imgs])
        targets = torch.load(f'./data/raw/train_target_{i}.pt')
        train_targets = torch.cat([train_targets, targets])
    
    torch.save(normalize(train_images), f'./data/processed/train_images.pt')
    torch.save(train_targets, f'./data/processed/train_target.pt')

    test_images = torch.load('./data/raw/test_images.pt')
    torch.save(normalize(test_images), './data/processed/test_images.pt')
    shutil.copy('./data/raw/test_target.pt','./data/processed/test_target.pt')

    print('Dataset Processed!')