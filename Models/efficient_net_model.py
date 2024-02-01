import torch
import torchvision
import torchinfo
from torchinfo import summary
from torch.hub import load_state_dict_from_url

def get_efficient_net_model_weights():
    """
    Gets the EfficientNet 5 model weights
    """
    weights = torchvision.models.EfficientNet_B5_Weights.DEFAULT
    return weights

def get_model_transforms(weights):
    """
    Returns the models proper data transforms
    """
    efficient_net_transforms = weights.transforms()
    return efficient_net_transforms

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

def get_model_summary(model):
    """
    Give a summary of the model

    Args:
        model (nn.Module): the model that we want to the get the summary of

    Returns:
        Summary of the model
    """
    return summary(model=model,
                    input_size=(32, 3, 500, 500),
                    verbose= 0,
                    col_names=["input_size", "output_size", "num_params", "trainable"],
                    col_width=20,
                    row_settings=["var_names"]
    )

def update_model(model, class_names, device):
    """
    Updates the classfier layers of the model for transfer
    learning on new data

    Args:
        model (nn.Module): the model that we wish to update
        class_names (List[str]): list of the class names
        device (str): the type of the device we're working on 'CPU' or 'Cuda'

    Returns:

        nn.Module: the updated model
    """
    for param in model.features.parameters():
        param.requires_grad = False
    output_shape = len(class_names)

    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=2048,
                        out_features=output_shape,
                        bias=True)).to(device)
    return model