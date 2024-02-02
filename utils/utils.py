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

def visualize_training_accuracy(metrics, num_epochs):
    sns.lineplot(y= metrics["train_acc"], x= [i for i in range(num_epochs)], label= "train_acc")
    sns.lineplot(y= metrics["test_acc"], x= [i for i in range(num_epochs)], label= "test_acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

def visualize_training_loss(metrics, num_epochs):
    sns.lineplot(y= metrics["train_acc"], x= [i for i in range(num_epochs)], label= "train_acc")
    sns.lineplot(y= metrics["test_acc"], x= [i for i in range(num_epochs)], label= "test_acc")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    

def log_tensorboard(writer ,train_loss, train_acc, test_loss, test_acc, epoch):
    """
    We store all the metrics into the tensorboard

    Args:
        writer: the summarywrite that we'll use to store and monitor the metrics
        train_loss (float): the training loss of the epoch
        train_acc (float): the training accuracy of the epoch
        test_loss (float): the test loss of the epoch
        test_acc (float): the test accuracy of the epoch
        epoch (int): the current epoch
    """
    writer.add_scalar("Train_Loss", train_loss, epoch)
    writer.add_scalar("Train_Accuracy", train_acc)
    writer.add_scalar('Test_Accuracy', test_acc, epoch)
    writer.add_scalar('Test_loss', test_loss, epoch)