import torch

def get_miscellanous_variables(train_data):
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    return class_names, class_dict

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device