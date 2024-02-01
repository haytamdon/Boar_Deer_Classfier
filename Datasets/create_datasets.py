
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

def get_miscellanous_variables(train_data):
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    return class_names, class_dict

def create_dataloaders(train_data, test_data):
    train_dataloader = DataLoader(dataset=train_data,
                                    batch_size=32,
                                    num_workers=2,
                                    shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                batch_size=32,
                                num_workers=2 ,
                                shuffle=False)

    return train_dataloader, test_dataloader

