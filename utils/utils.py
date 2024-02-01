import torch
import torchinfo
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns

def get_miscellanous_variables(train_data):
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    return class_names, class_dict

def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device

def model_summary(model, tensor_size):
    return summary(model, input_size=tensor_size)

def visualize_training_metrics(metrics, num_epochs):
    sns.lineplot(y= metrics["train_acc"], x= [i for i in range(num_epochs)], label= "train_acc")
    sns.lineplot(y= metrics["test_acc"], x= [i for i in range(num_epochs)], label= "test_acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")