import shutil
import os
import torch

def normalize(images):
    mean = images.mean()
    std = images.std()

    return (images - mean) / std

if __name__ == '__main__':
    train_images = torch.tensor([])
    train_targets = torch.tensor([], dtype=int)
    for file in os.listdir('data/raw/'):
        if 'train_images_' in file:
            imgs = torch.load(os.path.join('data/raw',file))
            train_images = torch.cat([train_images, imgs])
        elif 'train_target_' in file:
            targets = torch.load(os.path.join('data/raw',file))
            train_targets = torch.cat([train_targets, targets])
    
    torch.save(normalize(train_images), f'./data/processed/train_images.pt')
    torch.save(train_targets, f'./data/processed/train_target.pt')

    test_images = torch.load('./data/raw/test_images.pt')
    torch.save(normalize(test_images), './data/processed/test_images.pt')
    shutil.copy('./data/raw/test_target.pt','./data/processed/test_target.pt')

    print('Dataset Processed!')