import torch
import torchinfo
from torchinfo import summary
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

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
    
def compute_f1_score(model, test_dataloader, device):
    model.eval()

    # Initializing empty lists to store true labels and model predictions
    all_true_labels = []
    all_predictions = []

    for input_data, target in test_dataloader:
        # Moving the data to the device the model is on
        input_data, target = input_data.to(device), target.to(device)

        # Forward pass to get the model output
        with torch.no_grad():
            output = model(input_data)

        predictions = torch.argmax(output, dim=1).cpu().numpy()

        # Appending true labels and predictions to the lists
        all_true_labels.extend(target.cpu().numpy())
        all_predictions.extend(predictions)

    # Calculating F1 score
    f1 = f1_score(all_true_labels, all_predictions, average='macro')
    return f1, all_true_labels, all_predictions

def compute_confusion_matrix(all_true_labels, all_predictions):
    conf_matrix = confusion_matrix(all_true_labels, all_predictions)
    return conf_matrix

def plot_confusion_matrix(conf_matrix):
    # Ploting confusion matrix using seaborn and matplotlib
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=range(2), yticklabels=range(2))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()