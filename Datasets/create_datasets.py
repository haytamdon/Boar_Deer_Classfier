
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.data import Dataset

def create_datasets(train_dir, test_dir, train_transform, test_transform):
    train_data = datasets.ImageFolder(root=train_dir,
                                        transform=train_transform,
                                        target_transform=None)

    test_data = datasets.ImageFolder(root=test_dir,
                                    transform=test_transform)

    return train_data, test_data

def create_dataloaders(train_data, test_data, batch_size):
    train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=batch_size,
                                    num_workers=2,
                                    shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=batch_size,
                                num_workers=2 ,
                                shuffle=False)

    return train_dataloader, test_dataloader

